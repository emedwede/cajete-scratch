set(SOURCE_FILES cell_list1D.cpp)
set(EXEC cell_list1D)

add_executable(${EXEC} ${SOURCE_FILES})

target_link_libraries(${EXEC} Cabana::cabanacore)

