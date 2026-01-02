#include "network.hpp"
#include <fstream>
#include <iostream>
#include <vector>
// #include <iomanip>

#include "json.hpp"
using json = nlohmann::json;

/*
NetworkConfig netconf{1, {{2}, {5}, {3}}};
Network network{netconf};

int main() {
  std::cout << std::fixed << std::setprecision(4);
  network.debug();
}
*/

// TODO: implement NN as discussed using ngram input format

std::string DE_PATH = "./data/de.txt";
std::string FR_PATH = "./data/fr.txt";

std::vector<std::string> LANG = {DE_PATH, FR_PATH};

size_t biN = 275, triN = 225;
NGramConfig ngramconf(biN, triN);

int main() {
  top_ngrams(LANG, ngramconf);
  std::string NGRAMS_PATH = "out/ngrams.json";
  NGramConfig ngram_config = load_ngrams(NGRAMS_PATH);
  size_t n = ngram_config.features;
}
