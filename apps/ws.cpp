#include "server_ws.hpp"
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include <queue>
#include <condition_variable>
#include <mutex>
#include <signal.h>

using namespace rapidjson;

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
                Document d;
                d.Parse(message.c_str());

                // FIXME: Remove echo
                d["a"] = d["a"].GetInt() + 1;
                send(d);
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

        void send(const Document& d) {
            StringBuffer buffer;
            Writer<StringBuffer> writer(buffer);
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

