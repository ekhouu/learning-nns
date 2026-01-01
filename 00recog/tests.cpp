#include "network.hpp"
#include <algorithm>
#include <fstream>
#include <vector>
// #include <iomanip>
#include <map>

#include "json.hpp"
using json = nlohmann::json;

/* testing network
 *
 * note: every neuron on one layer is connected to every neuron of the next
 * layer, we store network as astd::vector<std::pair<std::string, int>>
get_top_k(const std::map<std::string, int> &ngrams, int k)
std::map<std::string, int>
combine_ngrams(const std::vector<std::map<std::string, int>> &ngram_maps)
int top_ngrams(std::vector<std::string> &paths, int k)n array
 *
 */

/*
NetworkConfig netconf{1, {{2}, {5}, {3}}};
Network network{netconf};

int main() {
  std::cout << std::fixed << std::setprecision(4);
  network.debug();
}
*/

// n = ngram size
/* std::map<std::string, int> extract_ngrams(const std::string &filename, int n)
{ std::ifstream file(filename); std::map<std::string, int> ngrams; std::string
line;

  // TODO: count all words in file & add parameter to ngram picking
  while (std::getline(file, line)) {
    line.erase(std::remove_if(line.begin(), line.end(),
                              [](char c) { return (unsigned char)c > 127; }),
               line.end());
    for (size_t i = 0; i + n <= line.length(); i++) {
      std::string ngram = line.substr(i, n);
      ngrams[ngram]++;
    }
  }
  return ngrams;
}

std::vector<std::pair<std::string, int>>
get_top_k(const std::map<std::string, int> &ngrams, int k) {
  std::vector<std::pair<std::string, int>> sorted(ngrams.begin(), ngrams.end());
  std::partial_sort(
      sorted.begin(), sorted.begin() + k, sorted.end(),
      [](const auto &a, const auto &b) { return a.second > b.second; });
  sorted.resize(k);
  return sorted;
}

std::map<std::string, int>
combine_ngrams(const std::vector<std::map<std::string, int>> &ngram_maps) {
  std::map<std::string, int> out;
  for (auto &map : ngram_maps) {
    for (const auto &[key, value] : map) {
      out[key] += value;
    }
  }

  return out;
}

int top_ngrams(std::vector<std::string> &paths, int k) {
  std::vector<std::map<std::string, int>> bi, tri;
  for (auto &p : paths) {
    bi.push_back(extract_ngrams(p, 2));
    tri.push_back(extract_ngrams(p, 3));
  }

  std::map<std::string, int> bigrams = combine_ngrams(bi);
  std::map<std::string, int> trigrams = combine_ngrams(tri);

  json ngrams_data;
  ngrams_data["bigrams"] = get_top_k(bigrams, k);
  ngrams_data["trigrams"] = get_top_k(trigrams, k);
  std::ofstream("out/ngrams.json") << ngrams_data.dump(2);

  return 1;
}
*/

int K_VALUE = 125;
std::string DE_PATH = "./data/de.txt";
std::string FR_PATH = "./data/fr.txt";

std::vector<std::string> LANG = {DE_PATH, FR_PATH};
int main() { top_ngrams(LANG, K_VALUE); }
