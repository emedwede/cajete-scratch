#########################
# Eventually use CTEST!
#########################
#LIST(APPEND TESTS example)

#foreach(test ${TESTS})
#        add_executable(${test} ./${test}.cpp)
#        add_test(NAME ${test} COMMAND ./${test})
#endforeach(test)


## For now this should work, see catch2 doc later to make it
## more sophisticated

add_executable(tests tests-main.cpp grid-test.cpp simple-cell-list-test.cpp)
add_executable(graph_test tests-main.cpp graph-test.cpp)
add_executable(alloc_test tests-main.cpp alloc-test.cpp)
add_executable(resource_test tests-main.cpp resource-test.cpp)
add_executable(node_list_test tests-main.cpp node-list-test.cpp)

target_link_libraries(tests simplecajete)
target_link_libraries(tests Cabana::cabanacore)

target_link_libraries(graph_test simplecajete)
target_link_libraries(graph_test Cabana::cabanacore)

target_link_libraries(alloc_test simplecajete)
target_link_libraries(alloc_test Cabana::cabanacore)

target_link_libraries(resource_test simplecajete)
target_link_libraries(resource_test Cabana::cabanacore)

target_link_libraries(node_list_test simplecajete)
target_link_libraries(node_list_test Cabana::cabanacore)
