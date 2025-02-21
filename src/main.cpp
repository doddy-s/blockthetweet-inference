#include <iostream>
#include "../libs/http/httplib.h"
#include "../libs/nlohmann/json.hpp"
#include <torch/script.h>
#include <libstemmer.h>
#include <filesystem>
#include <chrono>
#include "../libs/xxhash/xxhash.h"
#include "../libs/cxxopts/cxxopts.hpp"

#define author "doddy-s"
#define version "v0.1"
#define appName "BlockTheTweet Inference"

// Global variables
torch::jit::script::Module MODEL; // Single model
nlohmann::json WORD_INDEX; // Word index for tokenization
sb_stemmer* STEMMER; // Stemmer for preprocessing text

// Struct to hold prediction results
struct Prediction {
    std::string text;
    uint64_t text_hash;
    float confidence;
    long long nanosecond;

    std::string toResponseData() {
        return nlohmann::json{
            {"text_hash", text_hash},
            {"text", text},
            {"confidence", confidence},
            {"nanosecond", nanosecond}
        }.dump();
    }
};

// Utility function to construct a JSON response
std::string constructResponse(const int& statusCode, const std::string& message, nlohmann::json data = nullptr) {
    auto response = nlohmann::json{
        {"statusCode", statusCode},
        {"message", message}
    };

    if (data != nullptr) {
        response["data"] = data;
    }

    return response.dump();
}

// Function to stem a word using the stemmer
std::string stemWord(const std::string& word) {
    const sb_symbol* stemmed = sb_stemmer_stem(STEMMER, (const sb_symbol*)word.c_str(), word.size());
    int stemmed_length = sb_stemmer_length(STEMMER);
    return std::string(reinterpret_cast<const char*>(stemmed), stemmed_length);
}

// Function to tokenize text
std::vector<int64_t> tokenizeText(const std::string& text, size_t max_length) {
    std::vector<int64_t> tokenized_text;

    // Tokenize the input text into words
    std::istringstream stream(text);
    std::string word;

    while (stream >> word) {
        // Convert word to lowercase to ensure case-insensitivity
        for (auto& c : word) c = tolower(c);

        // Stem the word
        word = stemWord(word);

        // Check if the word exists in the word_index, otherwise assign default value (e.g., 0)
        if (WORD_INDEX.find(word) != WORD_INDEX.end()) {
            tokenized_text.push_back(WORD_INDEX[word]);
        }
        else {
            tokenized_text.push_back(0);  // Default index for unknown words
        }
    }

    // Truncate or pad the vector to match max_length
    if (tokenized_text.size() > max_length) {
        tokenized_text.resize(max_length);
    }
    else if (tokenized_text.size() < max_length) {
        tokenized_text.resize(max_length, 0);  // Pad with 0
    }

    return tokenized_text;
}

// Function to predict text using the loaded model
bool predictText(const std::string& text, Prediction& prediction) {
    try {
        // Tokenize the input text
        std::vector<int64_t> input_data = tokenizeText(text, 34);
        torch::Tensor input_tensor = torch::tensor(input_data, torch::dtype(torch::kLong)).unsqueeze(0);
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);

        // Perform prediction and measure time
        auto beginOfPredictTime = std::chrono::high_resolution_clock::now();
        at::Tensor output = MODEL.forward(inputs).toTensor();
        auto endOfPredictTime = std::chrono::high_resolution_clock::now();
        auto predictTime = std::chrono::duration_cast<std::chrono::nanoseconds>(endOfPredictTime - beginOfPredictTime).count();

        // Store prediction results
        prediction.text = text;
        prediction.text_hash = XXH64(text.c_str(), text.size(), 0);
        prediction.confidence = output.item<float>();
        prediction.nanosecond = predictTime;

        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Prediction error: " << e.what() << std::endl;
        return false;
    }
}

// Controller for handling text classification requests
void postClassifyText(const httplib::Request& req, httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    try {
        nlohmann::json reqBody;
        try {
            reqBody = nlohmann::json::parse(req.body);
        }
        catch (const std::exception& e) {
            res.status = 400;
            res.set_content(constructResponse(400, "Bad Request"), "application/json");
            return;
        }

        std::string text = reqBody["text"];

        Prediction prediction;

        if (!predictText(text, prediction)) {
            res.status = 500;
            res.set_content(constructResponse(500, "Internal Server Error"), "application/json");
            return;
        }

        res.status = 200;
        res.set_content(prediction.toResponseData(), "application/json");
    }
    catch (const std::exception& e) {
        std::cerr << "Caught standard exception: " << e.what() << std::endl;
        res.status = 500;
        res.set_content(constructResponse(500, "Internal Server Error"), "application/json");
        return;
    }
}

// Controller for providing application information
void getInformations(const httplib::Request&, httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    auto data = nlohmann::json{
        {"author", author},
        {"version", version},
        {"appName", appName}
    };
    res.status = 200;
    res.set_content(constructResponse(200, "success", data), "application/json");
}

// Function to attach routes to the server
void attachRoutes(httplib::Server& server) {
    server.Options(".*", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
        res.set_header("Access-Control-Max-Age", "86400");
        res.status = 204; // No content
        });

    server.Get("/", getInformations);
    server.Post("/", postClassifyText);
}

// Main function
int main(int argc, char* argv[]) {
    // Parse command-line arguments
    cxxopts::Options options("BlockTheTweet", "A server for text classification using PyTorch models.");
    options.add_options()
        ("m,model-path", "Path to the model file", cxxopts::value<std::string>()->default_value("./resources/model.pt"))
        ("w,word-index-path", "Path to word index JSON file", cxxopts::value<std::string>()->default_value("./resources/word_index.json"))
        ("s,stemmer-lang", "Stemmer language", cxxopts::value<std::string>()->default_value("english"))
        ("p,port", "Port to run the server on", cxxopts::value<int>()->default_value("3000"))
        ("h,help", "Print usage");

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    std::string modelPath = result["model-path"].as<std::string>();
    std::string wordIndexPath = result["word-index-path"].as<std::string>();
    std::string stemmerLang = result["stemmer-lang"].as<std::string>();
    int port = result["port"].as<int>();

    httplib::Server server;

    try {
        // Load the model
        MODEL = torch::jit::load(modelPath);
        std::cout << "Loaded model from: " << modelPath << std::endl;

        // Load word index for tokenization
        std::ifstream f(wordIndexPath);
        WORD_INDEX = nlohmann::json::parse(f);
        std::cout << "Loaded word index from: " << wordIndexPath << std::endl;

        // Initialize the stemmer
        STEMMER = sb_stemmer_new(stemmerLang.c_str(), nullptr);
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
        return -1;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    // Attach routes to the server
    attachRoutes(server);

    // Start the server
    std::cout << "BlockTheTweet Server Is Running At Port " << port << "\n";
    server.listen("0.0.0.0", port);
}