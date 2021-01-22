#set(SOURCE_FILES cell_list_simple.cpp)
#set(EXEC cell_list_simple)
file(GLOB HEADER_FILES "*.hpp")
file(GLOB SOURCE_FILES "*.cpp")

#set(LIB_NAME simplecajete)
add_library(simplecajete ${SOURCE_FILES} ${HEADER_FILES})
target_link_libraries(simplecajete Cabana::cabanacore)
