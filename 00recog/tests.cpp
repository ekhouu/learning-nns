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

/* int K_VALUE = 125;
std::string DE_PATH = "./data/de.txt";
std::string FR_PATH = "./data/fr.txt";

std::vector<std::string> LANG = {DE_PATH, FR_PATH};

size_t biN = 275, triN = 225;
NGramConfig ngramconf(biN, triN);
int main() { top_ngrams(LANG, ngramconf); }
*/

// for now, we're using a frequency vector

// TODO: consider reworking this loading thing cuz it's kind of ass
NGramConfig load_ngrams(std::string ngrams_json) {

  std::map<std::string, size_t> conv;
  std::ifstream file(ngrams_json);
  json j = json::parse(file);

  size_t bk = 0, tk;

  if (j.contains("loaded")) {
    conv = j["loaded"]["conv"];
    bk = j["loaded"]["n_bigrams"];
    tk = j["loaded"]["n_trigrams"];
  } else {
    for (auto &b : j["bigrams"]) {
      conv[b[0]] = bk;
      bk++;
    }

    tk = bk;

    for (auto &t : j["trigrams"]) {
      conv[t[0]] = tk;
      tk++;
    }

    tk = tk - bk;

    j["loaded"]["conv"] = conv;
    j["loaded"]["n_bigrams"] = bk;
    j["loaded"]["n_trigrams"] = tk;

    std::ofstream(ngrams_json) << j.dump();
  }

  NGramConfig ngram_config(bk, tk);
  ngram_config.conv = conv;
  return ngram_config;
}

void found_incre(std::string key, std::map<std::string, size_t> &map,
                 std::vector<double> &vec) {
  auto it = map.find(key);
  if (it != map.end()) {
    vec[it->second]++;
  }
}

std::vector<double> ngramify(std::string s, NGramConfig &ngram_config) {
  std::vector<double> ngram_vec(ngram_config.features, 0);

  size_t l = s.size();

  if (l > 2) {
    for (size_t i = 0; i < l - 2; i++) {
      found_incre({s[i], s[i + 1]}, ngram_config.conv, ngram_vec);
      found_incre({s[i], s[i + 1], s[i + 2]}, ngram_config.conv, ngram_vec);
    }
  }

  if (l > 1) {
    found_incre({s[l - 2], s[l - 1]}, ngram_config.conv, ngram_vec);
  }

  return ngram_vec;
}

// TODO: create match_ngrams to take ngrams and format and output filled on

// TODO: implement NN as discussed using ngram input format

std::string DE_PATH = "./data/de.txt";
std::string FR_PATH = "./data/fr.txt";

std::vector<std::string> LANG = {DE_PATH, FR_PATH};

size_t biN = 275, triN = 225;
NGramConfig ngramconf(biN, triN);

int main() {
  // top_ngrams(LANG, ngramconf);
  std::string NGRAMS_PATH = "out/ngrams.json";
  NGramConfig ngram_config = load_ngrams(NGRAMS_PATH);
  size_t n = ngram_config.features;
}
