#include <Cabana_Core.hpp>

class ResourceBase {
    public:
        
        Kokkos::View<int*> data;

        KOKKOS_FUNCTION
        int allocate(int n) const {
            return do_allocate(n);
        }
        
        KOKKOS_FUNCTION
        void insert(int n, int val) const {
            do_insert(n, val);
        }

        void set_data(int c) {
            do_set_data(c);
        }

        KOKKOS_FUNCTION
        int at(int n) const {
            return do_at(n);
        }

        KOKKOS_FUNCTION
        virtual ~ResourceBase() {}

    private:
        KOKKOS_FUNCTION
        virtual int do_allocate(int n) const = 0;

        KOKKOS_FUNCTION
        virtual void do_insert(int n, int val) const = 0;

        virtual void do_set_data(int c) = 0;

        KOKKOS_FUNCTION
        virtual int do_at(int n) const = 0;
    
    protected:
        int value;
};

class ChildResource : public ResourceBase {
    public:
        KOKKOS_FUNCTION
        ChildResource(int c) {
            printf("Constructed child with value %d\n", c);
        }
       
        int member;
        
    private:
                
        KOKKOS_FUNCTION
        virtual int do_at(int n) const override {
            return data(n);
        }

        virtual void do_set_data(int c) override {
            data = Kokkos::View<int*>("name", c);
        }

        KOKKOS_FUNCTION
        virtual int do_allocate(int n) const override {
            printf("Child allocate %d\n", n);
            return 1;
        }

        KOKKOS_FUNCTION
        virtual void do_insert(int n, int val) const override {
            data(n) = val;
        }
};

struct Base {

    KOKKOS_FUNCTION
    Base() { printf("Base Constructor\n"); }

    virtual void build_data(int c) = 0;

    KOKKOS_FUNCTION
    virtual void set_value(int input) = 0;
    KOKKOS_FUNCTION
    virtual ~Base() { printf("Base Destructor\n"); }
};

struct Child : Base {
    
    Kokkos::View<int*> data;
    int value;

    KOKKOS_FUNCTION
    Child() { 
        printf("Child contructor\n");
    }
    
    virtual void build_data(int c) override {
        data = Kokkos::View<int*>("name", c);
    }

    KOKKOS_FUNCTION
    virtual void set_value(int input) override {
        value = input;
    } 
};

//There is a difference between virtual functions that control execution on the host
//and those that are used in kernel!

int main (int argc, char* argv[]) {
    Kokkos::ScopeGuard scope_guard(argc, argv);
     
    #ifdef KOKKOS_ENABLE_CUDA
    using DeviceType = Kokkos::CudaUVMSpace;
    #else
    using DeviceType = Kokkos::HostSpace;
    #endif

    ResourceBase* deviceChildResource = 
        (ResourceBase*)Kokkos::kokkos_malloc<DeviceType>(sizeof(ChildResource));
  
    Child* deviceChild = (Child*)Kokkos::kokkos_malloc<DeviceType>(sizeof(Child));
    //ChildResource* deviceChildResource = 
    //    (ChildResource*)Kokkos::kokkos_malloc<DeviceType>(sizeof(ChildResource));
    Child* hostChild = (Child*)Kokkos::kokkos_malloc<Kokkos::HostSpace>(sizeof(Child));

    Kokkos::parallel_for("Create", 1, KOKKOS_LAMBDA(const int&) {
        new ((ChildResource*)deviceChildResource) ChildResource(10);
        new ((Child*)deviceChild) Child();
    });
    
    int* ptr = (int*) Kokkos::kokkos_malloc<DeviceType>(sizeof(int)*3);
    Kokkos::View<int*, DeviceType> test(ptr, 3);
    for(auto i = 0; i < 3; i++) {
        test(i) = i+3;
        std::cout << test(i) << " ";
    } std::cout << std::endl;

    Kokkos::View<Child*> deviceChild_v(deviceChild, 1);
    Kokkos::View<Child*, Kokkos::HostSpace> hostChild_v(hostChild, 1);
    
    
    Kokkos::deep_copy(hostChild_v, deviceChild_v);
    //hostChild_v(0).build_data(10);
    Kokkos::deep_copy(deviceChild_v, hostChild_v);

    //deviceChildResource->data = Kokkos::View<int*>("name", 10);
    
    ((ChildResource*)deviceChildResource)->member = 2;
    //deviceChildResource->set_data(10);
    //deviceChildResource->allocate(1); 
    Kokkos::parallel_for("Virtual Dispatch", 10, KOKKOS_LAMBDA(const int i) {
        //deviceChildResource->allocate(1);
        //deviceChildResource->insert(i, i*2);
    }); 
    Kokkos::fence();
    //deviceChild->build_data(10);
    
    for(auto i = 0; i < 10; i++) {
        //std::cout << deviceChid
    }
    Kokkos::parallel_for("Destroy", 1, KOKKOS_LAMBDA(const int&) {
        //deviceChildResource->~ChildResource();
        deviceChildResource->~ResourceBase();
        deviceChild->~Child();
    });

    Kokkos::kokkos_free(deviceChildResource);
    Kokkos::kokkos_free(deviceChild);
    Kokkos::kokkos_free(hostChild);
    return 0;
}
