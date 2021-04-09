#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>
#include <string>
#include <iostream>

template<typename T>
struct kokkos_vector {
 
    using ViewType_d = Kokkos::View<T*>;
    using ViewType_h = typename ViewType_d::HostMirror;
    using count_t_d = Kokkos::View<size_t>;
    using count_t_h = typename count_t_d::HostMirror;


    ViewType_d data_d;
    ViewType_h data_h;
    count_t_d size_d;
    count_t_h size_h;
    count_t_d capacity_d; 
    count_t_h capacity_h;

    //set aside a capacity and an initial size of zero
    kokkos_vector(std::string _name = "vector", size_t _capacity = 0) 
        : data_d(_name, _capacity)
        , data_h(_name, _capacity)
        , size_d("Size on Device")
        , size_h("Size on Host")
        , capacity_d("Capacity on Device")
        , capacity_h("Capacity on Host")
    {
        Kokkos::deep_copy(size_d, 0);
        Kokkos::deep_copy(size_h, 0);
        Kokkos::deep_copy(capacity_d, _capacity);
        Kokkos::deep_copy(capacity_h, _capacity);
    }
    
    void sync_host() {
        Kokkos::deep_copy(data_h, data_d);
        Kokkos::deep_copy(size_h, size_d);
        Kokkos::deep_copy(capacity_h, capacity_d);
    }

    void sync_device() {
        Kokkos::deep_copy(data_d, data_h);
        Kokkos::deep_copy(size_d, size_h);
        Kokkos::deep_copy(capacity_d, capacity_h);
    }

    KOKKOS_INLINE_FUNCTION
    void push_device(T val) const {
        int c = Kokkos::atomic_fetch_add( &size_d(), 1 );
        if(c >= capacity_d()) {
            printf("Out of memory error, device push failed!\n");
            Kokkos::atomic_decrement(&size_d());
        } else {
            data_d(c) = val;
        }

    }
    
    KOKKOS_INLINE_FUNCTION
    void push_host(T val) const {
        int c = Kokkos::atomic_fetch_add( &size_h(), 1 );
        if(c >= capacity_h()) {
            printf("Out of memory error, host push failed!\n");
            Kokkos::atomic_decrement(&size_h());
        } else {
            data_h(c) = val;
        }
    }
    
    void reserve(size_t _capacity) {
        Kokkos::resize(data_d, _capacity);
        Kokkos::resize(data_h, _capacity);
        Kokkos::deep_copy(capacity_d, _capacity);
        Kokkos::deep_copy(capacity_h, _capacity);
    }
};

int main(int argc, char* argv[]) {
    
    Kokkos::ScopeGuard scope_guard(argc, argv);

    kokkos_vector<int> my_vec("my_vec", 10);

    std::cout << "Size: " << my_vec.size_h() << std::endl;
    std::cout << "Capacity: " << my_vec.data_h.size() << std::endl;

    size_t num_pushes = 16;
    std::cout << "\nAttempting " << num_pushes << " pushes\n";

    const auto& _my_vec = my_vec;
    Kokkos::parallel_for("Push Test", num_pushes, KOKKOS_LAMBDA(const int i) {
        _my_vec.push_device(i*2);
        //printf("%zu\n", _my_vec.size_d());
    });
    my_vec.sync_host();
    printf("\nSize: %zu\n", my_vec.size_h());
    
    my_vec.reserve(100);
    my_vec.sync_host();

    std::cout << "Capacity: " << my_vec.data_h.size() << std::endl;
   
    std::cout << "Values: ";
    for(auto i = 0; i < my_vec.size_h(); i++) {
        std::cout << my_vec.data_h(i) << " ";
    } std::cout << std::endl;

    return 0;
}
