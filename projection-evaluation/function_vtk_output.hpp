#ifndef _FUNCTION_VTK_OUTPUT
#define _FUNCTION_VTK_OUTPUT



#include <iostream>
#include <fstream>
#include <string>



template <typename T>
inline void write_vtk(
  T &field,
  std::string filename
){


  std::filebuf file_buf;
  file_buf.open(filename, std::ios::out);
  std::ostream outfile(&file_buf);

  field.WriteVtkHeader(outfile);
  
  auto expansions = field.GetExp();
  const int num_expansions = (*expansions).size();
  for(int ex=0 ; ex<num_expansions ; ex++){
    field.WriteVtkPieceHeader(outfile, ex);
    field.WriteVtkPieceData(outfile, ex);
    field.WriteVtkPieceFooter(outfile, ex);
  }

  field.WriteVtkFooter(outfile);

  file_buf.close();
}





#endif
