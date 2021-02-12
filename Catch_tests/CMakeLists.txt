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
target_link_libraries(tests simplecajete)
target_link_libraries(tests Cabana::cabanacore)
target_link_libraries(graph_test simplecajete)
target_link_libraries(graph_test Cabana::cabanacore)
