#ifndef PIC_VIS_HPP
#define PIC_VIS_HPP

#include <iostream>
#include <fstream>
#include <unordered_set>
#include <set>
#include "graph.hpp"

template<typename DeviceType>
class Visualizer {
	
	public:
		std::ofstream vis_file;

		void write_header(size_t num_nodes, std::string title) {
			std::stringstream sstm;

			sstm << title << ".vtk";
			std::string file_name = sstm.str();

			vis_file.open(file_name);

			vis_file << "# vtk DataFile Version 2.0" << std::endl;
			vis_file << "Unstructured Grid Example" << std::endl;
			vis_file << "ASCII" << std::endl;
			vis_file << "" << std::endl;
			vis_file << "DATASET UNSTRUCTURED_GRID" << std::endl;

			vis_file << "POINTS " << num_nodes << " float" << std::endl;
		}

        void write_graph_position(Cajete::Graph<DeviceType>& system) 
		{
			for( std::size_t idx = 0; idx != system.get_size(); ++idx ) 
			{
				float x = system.positions_h(idx, 0);
				float y = system.positions_h(idx, 1);
                float z = 0.0;
				vis_file << x << " " << y << " " << z << std::endl;
			}
		}

        int write_cells(Cajete::Graph<DeviceType>& system)
		{
			std::string delimiter = "::";
			std::unordered_set<std::string> links;

			int num_nodes = system.get_size();

			for (int p = 0; p < num_nodes; p++)
			{
				//Traverse the links
				for (int l = 0; l < 4; l++) 
				{
					int con = system.edges_h(p, l);
					if (con != -1)
					{
						int _min = std::min(p, con);
						int _max = std::max(p, con);
						links.insert(std::to_string(_min) + delimiter + std::to_string(_max));
					}
				}
			}
			
			int num_write_values = links.size() * 3;
			vis_file << "CELLS " << links.size() << " " << (int)(num_write_values) << std::endl;

			for(auto l : links)
			{
				//Split it
				auto br = l.find(delimiter);
				std::string token1 = l.substr(0, br);
				std::string token2 = l.substr(br+delimiter.size());

				vis_file << "2 " << token1 << " " << token2 << std::endl;
			}

			return links.size();
		}


		void write_cell_types(size_t num_cells)
		{
			vis_file << "CELL_TYPES " << num_cells  << std::endl;

			// 1 = point
			// 3 = line
			// 4 = poly line
			// 5 = triangle
			// more https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf
			
			for(size_t i = 0; i < num_cells; i++)
			{
				vis_file << "4" << std::endl;
			}
		}

		void pre_scalars(size_t num_nodes)
		{
			vis_file << "POINT_DATA " << num_nodes << std::endl;
		}

		void write_graph_property_header(std::string name, size_t num_nodes)
		{
			vis_file << "SCALARS " << name << " float 1" << std::endl;
			vis_file << "LOOKUP_TABLE default" << std::endl;
		}
        
		void write_graph_types(Cajete::Graph<DeviceType>& system)
		{
			for (std::size_t idx = 0; idx != system.get_size(); ++idx)
			{
				vis_file << system.nodes_h(idx) << std::endl;
			}
		}
        
		void write_graph_sp(Cajete::Graph<DeviceType>& system, size_t sn)
		{
			for(std::size_t idx = 0; idx != system.get_size(); ++idx)
			{
				vis_file << sn << std::endl;
			}
		}

		void finalize()
		{
			vis_file.close();
		}
        
		void write_vis(Cajete::Graph<DeviceType>& system, std::string name)
		{
			size_t num_nodes = system.get_size();
			
            write_header(num_nodes, name);

			write_graph_position(system);

			int num_cells = write_cells(system);
			write_cell_types(num_cells);

			pre_scalars(num_nodes);

			write_graph_property_header("node_type", num_nodes);
			write_graph_types(system);

			finalize();

		}

};

#endif //Visualizer
