#include "network.hpp"
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

int K_VALUE = 125;
std::string DE_PATH = "./data/de.txt";
std::string FR_PATH = "./data/fr.txt";

std::vector<std::string> LANG = {DE_PATH, FR_PATH};
int main() { top_ngrams(LANG, {275, 125}); }
