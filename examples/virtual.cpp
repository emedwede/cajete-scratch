#include <Cabana_Core.hpp>

class ResourceBase {
    public:
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
};

class ChildResource : public ResourceBase {
    public:
        KOKKOS_FUNCTION
        ChildResource(int c) {
            printf("Constructed child with value %d\n", c);
        }
        
    private:
        Kokkos::View<int*> data;
        
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

class Foo {
    protected:
        int val;

    public:
        KOKKOS_FUNCTION
        Foo() {val = 0;}

        KOKKOS_FUNCTION
        virtual int value() const = 0;
        //virtual int value() {return 0;}

        KOKKOS_FUNCTION
        virtual ~Foo() {}

};

class Foo_1 : public Foo {
    public:
        KOKKOS_FUNCTION
        Foo_1() { val = 1; }

        KOKKOS_FUNCTION
        virtual int value() const override { return val; }
};

class Foo_2 : public Foo {
    public:
        KOKKOS_FUNCTION
        Foo_2() { val = 2; }

        KOKKOS_FUNCTION
        int value() const { return val; }
};

int main (int argc, char* argv[]) {
    Kokkos::ScopeGuard scope_guard(argc, argv);
     
    #ifdef KOKKOS_ENABLE_CUDA
    using DeviceType = Kokkos::CudaUVMSpace;
    #else
    using DeviceType = Kokkos::HostSpace;
    #endif

    //ResourceBase* deviceChildResource = 
    //    (ResourceBase*)Kokkos::kokkos_malloc<DeviceType>(sizeof(ChildResource));
   
    ChildResource* deviceChildResource = 
        (ChildResource*)Kokkos::kokkos_malloc<DeviceType>(sizeof(ChildResource));
    
 
    Foo* f_1 = (Foo*)Kokkos::kokkos_malloc<DeviceType>(sizeof(Foo_1));
    Kokkos::parallel_for("Create", 1, KOKKOS_LAMBDA(const int&) {
        new ((Foo_1*)f_1) Foo_1();
        new ((ChildResource*)deviceChildResource) ChildResource(10);
    });

    deviceChildResource->set_data(10);
    
    Kokkos::parallel_for("Virtual Dispatch", 10, KOKKOS_LAMBDA(const int i) {
        //int a = f_1->value();
        //printf("Value: %d\n", a);
        //deviceChildResource->allocate(1);
        //deviceChildResource->insert(i, i*2);

    }); 
   
    for(auto i = 0; i < 10; i++) {
        //std::cout << deviceChid
    }
    Kokkos::parallel_for("Destroy", 1, KOKKOS_LAMBDA(const int&) {
        f_1->~Foo();
        deviceChildResource->~ChildResource();
        //deviceChildResource->~ResourceBase();
    });

    Kokkos::kokkos_free(f_1);
    Kokkos::kokkos_free(deviceChildResource);

    return 0;
}
