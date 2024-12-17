#include <iostream>
#include "../libs/http/httplib.h"
#include "../libs/nlohmann/json.hpp"
#include <sqlite3.h>
#include "../libs/xxhash/xxhash.h"
#include <torch/script.h>
#include <libstemmer.h>

#define author "doddy-s"
#define version "v0.1"
#define appName "BlockTheTweet Inference"

torch::jit::script::Module MODEL;
nlohmann::json_abi_v3_11_3::json WORD_INDEX;
sb_stemmer* STEMMER;

// BEGIN OF STRUCTS
struct Prediction {
    int id;
    std::string timestamp;
    uint64_t text_hash;
    std::string text;
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
// END OF STRUCTS

// BEGIN OF DATABASE
class Database {
private:
    sqlite3* db;

    bool executeSql(const std::string sql) {
        char* errorMessage;
        int rc = sqlite3_exec(db, sql.c_str(), nullptr, nullptr, &errorMessage);
        if (rc != SQLITE_OK) {
            std::cerr << "SQL error: " << errorMessage << std::endl;
            sqlite3_free(errorMessage);
            return false;
        }
        return true;
    }

public:
    Database(const std::string& dbPath) {
        // Open database
        int rc = sqlite3_open(dbPath.c_str(), &db);
        if (rc) {
            throw std::runtime_error("Cannot open database: " + std::string(sqlite3_errmsg(db)));
        }

        // Create errors table if not exists
        std::string createTableSQL = R"(
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                log_type TEXT,
                message TEXT
            );
            
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                text_hash INTEGER UNIQUE,
                text TEXT,
                confidence FLOAT,
                second INTEGER
            );
        )";

        if (!executeSql(createTableSQL)) {
            throw std::runtime_error("Failed to create tables");
        }
    }

    // bool writePrediction(const Prediction& pred) {
    //     // Create the SQL statement to insert data into the predictions table
    //     std::string sql = R"(
    //         INSERT INTO predictions (text_hash, text, confidence, second)
    //         VALUES (?, ?, ?, ?);
    //     )";

    //     // Prepare the SQL statement
    //     sqlite3_stmt* stmt;
    //     int rc = sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr);
    //     if (rc != SQLITE_OK) {
    //         std::cerr << "Failed to prepare statement: " << sqlite3_errmsg(db) << std::endl;
    //         return false;
    //     }

    //     // Bind the values from the Prediction struct to the prepared statement
    //     sqlite3_bind_int(stmt, 1, pred.text_hash); // Bind text_hash
    //     sqlite3_bind_text(stmt, 2, pred.text.c_str(), -1, SQLITE_STATIC);
    //     sqlite3_bind_double(stmt, 3, pred.confidence);
    //     sqlite3_bind_double(stmt, 4, pred.nanosecond);

    //     // Execute the SQL statement
    //     rc = sqlite3_step(stmt);
    //     if (rc != SQLITE_DONE) {
    //         std::cerr << "Execution failed: " << sqlite3_errmsg(db) << std::endl;
    //         sqlite3_finalize(stmt);
    //         return false;
    //     }

    //     // Finalize the statement to release resources
    //     sqlite3_finalize(stmt);
    //     return true;
    // }

};
Database* DATABASE;
// END OF DATABASE

// BEGIN OF UTILS
std::string constructResponse(const int& statusCode, const std::string& message) {
    auto response = nlohmann::json{
        {"statusCode", statusCode},
        {"message", message} };

    return response.dump();
}

std::string constructResponse(const int& statusCode, const std::string& message, nlohmann::json& data) {
    auto response = nlohmann::json{
        {"statusCode", statusCode},
        {"message", message} };

    if (data != NULL) {
        response["data"] = data;
    }

    return response.dump();
}

std::string stemWord(const std::string& word) {
    const sb_symbol* stemmed = sb_stemmer_stem(STEMMER, (const sb_symbol*)word.c_str(), word.size());

    int stemmed_length = sb_stemmer_length(STEMMER);

    return std::string(reinterpret_cast<const char*>(stemmed), stemmed_length);
}


std::vector<int64_t> tokenizeText(const std::string& text, size_t max_length) {
    std::vector<int64_t> tokenized_text;

    // Tokenize the input text into words
    std::istringstream stream(text);
    std::string word;

    while (stream >> word) {
        // Convert word to lowercase to ensure case-insensitivity (optional)
        for (auto& c : word) c = tolower(c);

        word = stemWord(word);

        // Check if the word exists in the word_index, otherwise assign default value (e.g., 0)
        if (WORD_INDEX.find(word) != WORD_INDEX.end()) {
            tokenized_text.push_back(WORD_INDEX[word]);
        }
        else {
            tokenized_text.push_back(0);  // Default index for unknown words
        }
    }

    // If the vector is longer than max_length, truncate it
    if (tokenized_text.size() > max_length) {
        tokenized_text.resize(max_length);
    }
    // If the vector is shorter than max_length, pad it with 0s
    else if (tokenized_text.size() < max_length) {
        tokenized_text.resize(max_length, 0);  // Pad with 0
    }

    for (size_t i = 0; i < 10 && i < tokenized_text.size(); ++i) {
        std::cout << tokenized_text[i] << " ";
    }
    std::cout << std::endl;

    return tokenized_text;
}

bool predictText(const std::string& text, Prediction& prediction) {
    try {
        std::vector<int64_t> input_data = tokenizeText(text, 295);
        torch::Tensor input_tensor = torch::tensor(input_data, torch::dtype(torch::kLong)).unsqueeze(0);
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);

        auto beginOfPredictTime = std::chrono::high_resolution_clock::now();
        at::Tensor output = MODEL.forward(inputs).toTensor();
        auto endOfPredictTime = std::chrono::high_resolution_clock::now();
        auto predictTime = std::chrono::duration_cast<std::chrono::nanoseconds>(endOfPredictTime - beginOfPredictTime).count();

        float confidence = output.item<float>();

        prediction.text = text;
        prediction.confidence = confidence;
        prediction.nanosecond = predictTime;

        return true;
    }
    catch (const std::exception& e) {
        return false;
    }
}
// END OF UTILS

// BEGIN OF CONTROLLER
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

        auto text_hash = XXH64(text.c_str(), text.size(), 0);
        prediction.text_hash = text_hash;

        predictText(text, prediction);
        // DATABASE->writePrediction(prediction);

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

void getInformations(const httplib::Request&, httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    auto data = nlohmann::json{ {"author", author}, {"version", version}, {"appName", appName} };
    res.status = 200;
    res.set_content(constructResponse(200, "success", data), "application/json");
}
// END OF CONTROLLER

// BEGIN OF ROUTES
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
// END OF ROUTES

// BEGIN OF MAIN
int main() {
    httplib::Server server;

    try {
        MODEL = torch::jit::load("./bilstm-en-683k.pt");

        std::ifstream f("./word-index.json");
        WORD_INDEX = nlohmann::json::parse(f);

        DATABASE = new Database("./block_the_tweet.sqlite");

        STEMMER = sb_stemmer_new("english", nullptr);
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
        return -1;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    attachRoutes(server);

    std::cout << "BlockTheTweet Server Is Running At Port 3000\n";
    server.listen("0.0.0.0", 3000);
}
// END OF MAIN