/********************************************************************************
  Copyright (C) 2019 - 2022 by the lifex authors.

  This file is part of lifex.

  lifex is free software; you can redistribute it and/or modify
  it under the terms of the GNU Lesser General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  lifex is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public License
  along with lifex.  If not, see <http://www.gnu.org/licenses/>.
********************************************************************************/

/**
 * @file
 *
 * @author Federica Botta <federica.botta@mail.polimi.it>.
 * @author Matteo Calafà <matteo.calafa@mail.polimi.it>.
 */

#ifndef DG_ERROR_PARSER_HPP_
#define DG_ERROR_PARSER_HPP_

#include <time.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace error_parser
{
  /// Return the today's date.
  std::string
  get_date()
  {
    char   date[100];
    time_t curr_time;
    time(&curr_time);
    tm curr_tm;
    localtime_r(&curr_time, &curr_tm);
    strftime(date, 100, "%D %T", &curr_tm);
    return date;
  }

  /// Create the datafile for the errors.
  void
  initialize_datafile(const unsigned int dim,
                      const std::string &filename,
                      const char *       solution_name)
  {
    std::ofstream outdata;
    outdata.open("errors_" + filename + "_" + std::to_string(dim) + "D_" +
                 solution_name + ".data");
    if (!outdata)
      {
        std::cerr << "Error: file could not be opened" << std::endl;
        exit(1);
      }

    const std::string date = get_date();

    outdata << "nref" << '\t' << "l_inf" << '\t' << "l_2" << '\t' << "h_1"
            << '\t' << "DG" << '\t' << "date" << std::endl;

    for (unsigned int i = 1; i <= 5; ++i)
      outdata << i << '\t' << "x" << '\t' << "x" << '\t' << "x" << '\t' << "x"
              << '\t' << date << std::endl;

    outdata.close();
  }

  /// Update the datafile with the new errors.
  void
  update_datafile(const unsigned int         dim,
                  const unsigned int         nref,
                  const std::string &        title,
                  const std::vector<double> &errors,
                  const char *               solution_name)
  {
    const std::string filename = "errors_" + title + "_" + std::to_string(dim) +
                                 "D_" + solution_name + ".data";

    if (!std::filesystem::exists(filename))
      initialize_datafile(dim, title, solution_name);

    std::ifstream indata;
    std::ofstream outdata;
    indata.open(filename);
    outdata.open("errors_tmp.data");

    if (!outdata || !indata)
      {
        std::cerr << "Error: file could not be opened" << std::endl;
        exit(1);
      }

    std::string line;
    std::getline(indata, line);

    const char n_ref_c = '0' + nref;

    outdata << line << std::endl;

    const std::string date = get_date();

    for (unsigned int i = 0; i < 10; ++i)
      {
        std::getline(indata, line);

        if (line[0] == n_ref_c)
          outdata << nref << '\t' << errors[0] << '\t' << errors[1] << '\t'
                  << errors[2] << '\t' << errors[3] << '\t' << date
                  << std::endl;
        else
          outdata << line << std::endl;
      }

    outdata.close();
    indata.close();

    std::ifstream indata2("errors_tmp.data");
    std::ofstream outdata2(filename);
    outdata2 << indata2.rdbuf();
    outdata2.close();
    indata2.close();

    if (remove("errors_tmp.data") != 0)
      {
        std::cerr << "Error in deleting file" << std::endl;
        exit(1);
      }
  }
} // namespace error_parser

#endif /* DG_ERROR_PARSER_HPP_*/
