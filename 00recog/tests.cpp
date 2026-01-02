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

// TODO: featurize
int featurize(std::vector<std::string> &filenames, NGramConfig &ngram_config,
              size_t max_batch, std::string batch_dir) {
  std::map<std::string, std::map<std::string, std::vector<double>>> data;
  std::string line;

  // i know batches can be found w/ math but im going to manually do it just to
  // be sure for now, will fix later
  size_t batch = 0, batches = 0;
  for (auto &filename : filenames) {
    std::ifstream file(filename);
    while (std::getline(file, line)) {
      line = standardize(line);
      data[filename][line] = ngramify(line, ngram_config);
      batch++;
      if (batch != 0 && batch % max_batch == 0) {
        json j;
        j["n_batch"] = max_batch;
        j["batch_n"] = batches++;
        j["data"] = data;
        data.clear();
        std::ofstream(batch_dir + std::to_string(batches) + "_batch.json")
            << j.dump();
      }
    }
    file.close();
  }

  if (batch % max_batch != 0) {
    json j;
    j["n_batch"] = batch % max_batch;
    j["batch_n"] = batch / max_batch;
    j["data"] = data;
    std::ofstream(batch_dir + std::to_string(batches) + "_batch.json")
        << j.dump();
  }

  return 1;
}

std::string DE_PATH = "./data/de.txt";
std::string FR_PATH = "./data/fr.txt";

std::vector<std::string> LANG = {DE_PATH, FR_PATH};

size_t biN = 275, triN = 225;
NGramConfig ngramconf(biN, triN);

int main() {
  // top_ngrams(LANG, ngramconf);
  std::string NGRAMS_PATH = "out/ngrams.json";
  NGramConfig ngram_config = load_ngrams(NGRAMS_PATH);
  // size_t n = ngram_config.features;

  featurize(LANG, ngram_config, 100000, "./out/batches/");
}
