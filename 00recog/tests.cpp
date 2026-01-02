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

std::string DE_PATH = "./data/de.txt";
std::string FR_PATH = "./data/fr.txt";

std::vector<std::string> LANG = {DE_PATH, FR_PATH};

size_t biN = 175, triN = 125;
NGramConfig ngramconf(biN, triN);

std::string FEATURIZED = "./out/batches";

int main() {
  top_ngrams(LANG, ngramconf);
  std::string NGRAMS_PATH = "out/ngrams.json";
  NGramConfig ngram_config = load_ngrams(NGRAMS_PATH);
  size_t n = ngram_config.features;

  featurize_dataset(LANG, ngram_config, 100000, FEATURIZED);

  NetworkConfig netconf{n, {{1000}, {500}, {2}}};
  Network network{netconf};
}
