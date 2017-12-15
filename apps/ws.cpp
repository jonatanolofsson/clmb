#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "server_ws.hpp"
#include <Eigen/Core>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <queue>
#include <signal.h>

namespace rj = rapidjson;

struct HaltException : public std::exception {};

void termination_handler(int);

template<typename T>
class Q {
    private:
        std::queue<T> queue;
        std::mutex queue_guard;
        std::condition_variable data_available;
        std::atomic_bool dying = false;

    public:
        void push(const T& obj) {
            std::unique_lock<std::mutex> l(queue_guard);
            queue.push(obj);
            data_available.notify_all();
        }

        T pop() {
            std::unique_lock<std::mutex> l(queue_guard);
            while(!dying && queue.empty()) { data_available.wait(l); }
            if(dying) {
                throw HaltException();
            }
            T r = queue.front();
            queue.pop();
            return r;
        }

        void kill() {
            if(dying) { return; }
            dying = true;
            notify_all();
        }

        void notify_all() {
            data_available.notify_all();
        }

        ~Q() {
            kill();
        }
};

struct ParseError : public std::exception {
    private:
        std::string message;

    public:
        explicit ParseError(const std::string& msg) : message(msg) {}

        virtual const char* what() const throw() {
            return message.c_str();
        }
};


Eigen::MatrixXd getMatrix(const rj::Value& mat) {
    Eigen::MatrixXd res;
    if(!mat.IsArray()) {
    }
    const rj::SizeType rows = mat.Size();
    if(rows == 0) {
        throw ParseError("No rows.");
    }
    if(!mat[0].IsArray()) {
        throw ParseError("Row is not array.");
    }
    const rj::SizeType cols = mat[0].Size();
    if(cols == 0) {
        throw ParseError("No cols.");
    }
    res.resize(rows, cols);
    for(rj::SizeType i = 0; i < rows; ++i) {
        if(!mat[i].IsArray()) {
            throw ParseError("Row is not array.");
        }
        if(mat[i].Size() != cols) {
            throw ParseError("Column size mismatch.");
        }
        const rj::Value& row = mat[i];
        for(rj::SizeType j = 0; j < cols; ++j) {
            res(i, j) = row[j].GetDouble();
        }
    }
    return res;
}

class Application {
    private:
        using WsServer = SimpleWeb::SocketServer<SimpleWeb::WS>;
        using QType = std::shared_ptr<std::string>;
        WsServer server;
        Q<QType> inputqueue;
        Q<QType> outputqueue;
        std::atomic_bool dying = false;
        std::thread reader_thread;
        std::thread writer_thread;
        std::thread websocket_thread;
        std::mutex death_guard;
        std::condition_variable death;

        void postman(const std::string& message) {
            try {
                rj::Document d;
                d.Parse(message.c_str());

                // FIXME: Remove
                if(d.HasMember("mat")) {
                        Eigen::MatrixXd mat = getMatrix(d["mat"]);
                        std::cout << "Got matrix: " << std::endl << mat << std::endl;
                }
            } catch(...) {
                std::cout << "Failed to parse as JSON: " << message << std::endl;
            }
        }

        void reader() {
            try {
                std::cout << "Reader thread started.""" << std::endl;
                while(!dying) {
                    std::string message = *inputqueue.pop();
                    postman(message);
                }
            } catch(HaltException&) {
            }
            std::cout << "Reader thread finished.""" << std::endl;
        }

        void writer() {
            try {
                std::cout << "Writer thread started.""" << std::endl;
                while(!dying) {
                    auto send_stream = std::make_shared<WsServer::SendStream>();
                    *send_stream << *outputqueue.pop();;
                    for(auto& a_connection : server.get_connections()) {
                        a_connection->send(send_stream);
                    }
                }
            } catch(HaltException&) {
            }
            std::cout << "Writer thread finished.""" << std::endl;
        }

        void websocket() {
            std::cout << "Websocket thread started.""" << std::endl;
            auto &endpoint = server.endpoint[".*"];

            endpoint.on_message = [this](std::shared_ptr<WsServer::Connection> /*connection*/, std::shared_ptr<WsServer::Message> message) {
                inputqueue.push(std::make_shared<std::string>(message->string()));
            };

            server.start();
            std::cout << "Websocket thread finished.""" << std::endl;
        }

    public:
        Application(unsigned short port=8080)
        : reader_thread(&Application::reader, this),
          writer_thread(&Application::writer, this),
          websocket_thread(&Application::websocket, this)
        {
            server.config.port = port;
            struct sigaction new_action;

            new_action.sa_handler = termination_handler;
            sigemptyset(&new_action.sa_mask);
            new_action.sa_flags = 0;

            sigaction (SIGINT, &new_action, NULL);
            sigaction (SIGHUP, &new_action, NULL);
            sigaction (SIGTERM, &new_action, NULL);
        }

        ~Application() {
            kill();
        }

        void kill() {
            if(dying) { return; }
            dying = true;
            server.stop();
            outputqueue.kill();
            inputqueue.kill();
            writer_thread.join();
            reader_thread.join();
            websocket_thread.join();
            death.notify_all();
        }

        bool dead() {
            return dying;
        }

        void send(const std::string& str) {
            if(dying) { return; }
            outputqueue.push(std::make_shared<std::string>(str));
        }

        void send(const rj::Document& d) {
            rj::StringBuffer buffer;
            rj::Writer<rj::StringBuffer> writer(buffer);
            d.Accept(writer);
            send(buffer.GetString());
        }

        void wait() {
            std::unique_lock<std::mutex> l(death_guard);
            while(!dying) { death.wait(l); }
        }
};

Application app;
void termination_handler(int) { app.kill(); }

int main() {
    app.wait();
}
