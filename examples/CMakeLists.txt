#set(SOURCE_FILES cell_list1D.cpp)
#set(EXEC cell_list1D)

#add_executable(${EXEC} ${SOURCE_FILES})

#target_link_libraries(${EXEC} Cabana::cabanacore)

add_executable(cell_list1D cell_list1D.cpp)
add_executable(hier_ssa hier_ssa.cpp)
add_executable(performance_tests performance_tests.cpp)

target_link_libraries(cell_list1D Cabana::cabanacore)
target_link_libraries(hier_ssa Cabana::cabanacore)
target_link_libraries(performance_tests Cabana::cabanacore)
