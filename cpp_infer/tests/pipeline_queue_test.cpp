#include "bounded_queue.hpp"
#include <future>
#include <iostream>
#include <stdexcept>
#include <thread>

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
int main() {
    using namespace std::chrono_literals;
    BoundedQueue<int> ordered(2);
    std::thread producer([&] { for (int i = 0; i < 10000; ++i) ordered.push(i); ordered.close(); });
    int item = -1, count = 0;
    while (ordered.pop(item)) require(item == count++, "FIFO order changed");
    producer.join();
    require(count == 10000 && ordered.finished(), "close did not drain queue");

    BoundedQueue<int> full(1);
    full.push(1);
    auto blockedProducer = std::async(std::launch::async, [&] { return full.push(2); });
    full.cancel();
    require(blockedProducer.wait_for(1s) == std::future_status::ready, "cancel did not wake producer");
    require(!blockedProducer.get() && !full.pop(item), "cancel retained work");

    BoundedQueue<int> empty(1);
    auto blockedConsumer = std::async(std::launch::async, [&] { int result; return empty.pop(result); });
    empty.close();
    require(blockedConsumer.wait_for(1s) == std::future_status::ready, "close did not wake consumer");
    require(!blockedConsumer.get(), "closed empty queue produced a value");

    BoundedQueue<int> latest(1);
    require(latest.pushLatest(1) == 0 && latest.pushLatest(2) == 1, "latest replacement count wrong");
    latest.close();
    require(latest.pop(item) && item == 2 && !latest.pop(item), "latest queue returned stale work");
    std::cout << "PASS: FIFO, bounded backpressure, cancellation, EOF and latest replacement\n";
}
