#include <iostream>
#include <vector>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/optional.hpp>

//#include "sentence_splitter.h"

using namespace std;

namespace po = boost::program_options;
namespace fs = boost::filesystem;
namespace pt = boost::property_tree;

po::variables_map ParseArg(int argc, char const *argv[]) {
  po::options_description options("Options");
  options.add_options()
    ("config", po::value<string>(), "Configuration file")
    ("help,h", "Help")
    ;

  po::variables_map values;
  po::store(po::parse_command_line(argc, argv, options), values);
  po::notify(values);

  if (values.count("help")) {
    cout << "Multi-Sense Embedding trainer" << endl;
  }

  return values;
}

pt::ptree ParseConfigFile(const string & config_path) {
  pt::ptree config;
  pt::read_ini(config_path, config);
  return config;
}

int main(int argc, char const *argv[]) {
  try {
    const auto args = ParseArg(argc, argv);
    const auto config = ParseConfigFile(args["config"].as<string>());
  } catch (exception & e) {
    cerr << e.what() << endl;
    exit(1);
  }

  return 0;
}
