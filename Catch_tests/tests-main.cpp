//#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_RUNNER //used for our custom main
#include "catch.hpp"
#include <Cabana_Core.hpp>

int main(int argc, char *argv[]) {

   //Kokkos::ScopeGuard scope_guard();
   
   //We should do something special here to initialze the arguments manually?
   Kokkos::initialize();
   printf("#On Kokkos execution space %s\n",typeid(Kokkos::DefaultExecutionSpace).name());
       
   int result = Catch::Session().run(argc, argv);
   Kokkos::finalize();
   return result;
}

