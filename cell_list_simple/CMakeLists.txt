set(SOURCE_FILES cell_list_simple.cpp)
set(EXEC cell_list_simple)

add_executable(${EXEC} ${SOURCE_FILES})
target_link_libraries(${EXEC} Cabana::cabanacore)
