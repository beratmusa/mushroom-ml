import React, { useEffect, useState } from "react";

function App() {
  const [currentScreen, setCurrentScreen] = useState("results"); // Ekran durumu
  const [results, setResults] = useState([]); // Model sonuçları
  const [inputData, setInputData] = useState(Array(21).fill("")); // Kullanıcı girdileri
  const [prediction, setPrediction] = useState(null); // Tahmin sonucu

  // Özellikler için seçenekler (örnek olarak 3 seçenek sağlanmıştır)
  const featureOptions = [
    {
      placeholder: " başlık şeklini seçiniz",
      options: ["çıngırak", "konik", "dışbükey", "düz", "topuzlu", "batık"],
    }, // Özellik 1 seçenekleri
    {
      placeholder: " başlık yüzeyi seçiniz",
      options: ["lifli", "oluklar", "pullu", "düz"],
    }, // Özellik 2 seçenekleri
    {
      placeholder: " başlık rengi seçiniz",
      options: [
        "kahverengi",
        "devetüyü",
        "tarçın",
        "gri",
        "yeşil",
        "pembe",
        "mor",
        "kırmızı",
        "beyaz",
        "yeşil",
      ],
    }, // Özellik 3 seçenekleri
    {
      placeholder: " morluklar seçiniz",
      options: ["var", "yok"],
    }, // Özellik 4 seçenekleri
    {
      placeholder: " koku seçiniz",
      options: [
        "badem",
        "anason",
        "kreozot",
        "şüpheli",
        "faul",
        "küflü",
        "hiçbiri",
        "keskin",
        "baharatlı",
      ],
    }, // Özellik 5 seçenekleri
    {
      placeholder: " solungaç eki seçiniz",
      options: ["ekli", "alçalan", "özgür", "çentikli"],
    }, // Özellik 6 seçenekleri
    {
      placeholder: " solungaç aralığı seçiniz",
      options: ["kapalı", "sık", "ayrık"],
    }, // Özellik 7 seçenekleri
    {
      placeholder: " solungaç boyutu seçiniz",
      options: ["geniş", "dar"],
    }, // Özellik 8 seçenekleri
    {
      placeholder: " solungaç rengi seçiniz",
      options: [
        "siyah",
        "kahverengi",
        "devetüyü",
        "çikolata",
        "gri",
        "yeşil",
        "turuncu",
        "pembe",
        "mor",
        "kırmızı",
        "beyaz",
        "yeşil",
      ],
    }, // Özellik 9 seçenekleri
    {
      placeholder: " sap şeklinde seçiniz",
      options: ["genişleme", "sivrilen"],
    }, // Özellik 10 seçenekleri
    {
      placeholder: " sap-kök seçiniz",
      options: ["soğanlı", "kulüp", "bardak", "eşit", "rizomorflar", "köklü"],
    }, // Özellik 11 seçenekleri
    {
      placeholder: " halka üstü sap yüzeyi seçiniz",
      options: ["lifli", "pullu", "ipeksi", "düz"],
    }, // Özellik 12 seçenekleri
    {
      placeholder: " halkanın altındaki sap yüzeyi seçiniz",
      options: ["lifli", "pullu", "ipeksi", "düz"],
    }, // Özellik 13 seçenekleri
    {
      placeholder: " halkanın üstündeki sap rengi seçiniz",
      options: [
        "kahverengi",
        "devetüyü",
        "tarçın",
        "gri",
        "turuncu",
        "pembe",
        "kırmızı",
        "beyaz",
        "sarı",
      ],
    }, // Özellik 14 seçenekleri
    {
      placeholder: " halka altı sap rengi seçiniz",
      options: [
        "kahverengi",
        "devetüyü",
        "tarçın",
        "gri",
        "turuncu",
        "pembe",
        "kırmızı",
        "beyaz",
        "sarı",
      ],
    }, // Özellik 15 seçenekleri
    {
      placeholder: " peçe rengi seçiniz",
      options: ["kahverengi", "turuncu", "beyaz", "sarı"],
    }, // Özellik 16 seçenekleri
    {
      placeholder: " halka numarası seçiniz",
      options: ["hiçbiri", "1", "2"],
    }, // Özellik 17 seçenekleri
    {
      placeholder: " halka tipi seçiniz",
      options: [
        "örümcek ağı",
        "geçici",
        "ışıl ışıl",
        "büyük",
        "hiçbiri",
        "kolye",
        "mantolama",
        "alan",
      ],
    }, // Özellik 18 seçenekleri
    {
      placeholder: " spor-baskı-rengi seçiniz",
      options: [
        "siyah",
        "kahverengi",
        "devetüyü",
        "çikolata",
        "yeşil",
        "turuncu",
        "mor",
        "beyaz",
        "sarı",
      ],
    }, // Özellik 19 seçenekleri
    {
      placeholder: " nüfus seçiniz",
      options: [
        "bolluk",
        "kümelenmiş",
        "çeşitli",
        "dağınık",
        "birçok",
        "yalnız",
      ],
    }, // Özellik 20 seçenekleri
    {
      placeholder: " doğal ortam seçiniz",
      options: [
        "otlar",
        "yapraklar",
        "çayırlar",
        "yollar",
        "kentsel",
        "atık",
        "orman",
      ],
    }, // Özellik 21 seçenekleri
    // Diğer özellikler
  ];

  // Sonuçları backend'den çek
  useEffect(() => {
    if (currentScreen === "results") {
      fetch("http://127.0.0.1:5000/results")
        .then((response) => response.json())
        .then((data) => setResults(data))
        .catch((error) => console.error("Sonuçlar alınamadı:", error));
    }
  }, [currentScreen]);

  // Yüzdelik dönüşüm fonksiyonu
  const toPercentage = (value, decimals = 2) => {
    if (typeof value === "number") {
      return `${(value * 100).toFixed(decimals)}%`;
    }
    return value; // Eğer sayı değilse olduğu gibi döndür
  };

  // Tahmin yap
  const handlePrediction = () => {
    fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ input: inputData.map((item) => item) }),
    })
      .then((response) => response.json())
      .then((data) => setPrediction(data))
      .catch((error) => console.error("Tahmin başarısız:", error));
  };

  return (
    <div className="container mx-auto mt-6">
      {/* Sayfa geçiş butonları */}
      <div className="flex justify-center space-x-4 mb-6">
        <button
          className={`px-4 py-2 rounded ${
            currentScreen === "results"
              ? "bg-blue-500 text-white"
              : "bg-gray-200"
          }`}
          onClick={() => setCurrentScreen("results")}
        >
          Model Sonuçları
        </button>
        <button
          className={`px-4 py-2 rounded ${
            currentScreen === "predict"
              ? "bg-blue-500 text-white"
              : "bg-gray-200"
          }`}
          onClick={() => setCurrentScreen("predict")}
        >
          Tahmin Yap
        </button>
      </div>

      {/* Sonuçlar ekranı */}
      {currentScreen === "results" && (
        <div>
          <h1 className="text-2xl font-bold text-center mb-4">
            Model Sonuçları
          </h1>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {results.map((step, index) => (
              <div key={index} className="border p-4 rounded-lg shadow-md">
                <h2 className="text-lg font-bold">{step.step}</h2>
                <div className="mt-2 space-y-4">
                  {step.results.map((result, idx) => (
                    <div key={idx} className="border p-2 rounded shadow-sm">
                      <h3 className="font-semibold">{result.model}</h3>
                      <ul className="text-sm mt-2">
                        {Object.entries(result.metrics).map(([key, value]) => (
                          <li key={key}>
                            {key !== "Confusion Matrix" ? (
                              <>
                                <span className="font-bold">{key}:</span>{" "}
                                {key.includes("Loss") ||
                                typeof value !== "number"
                                  ? value
                                  : toPercentage(value)}
                              </>
                            ) : (
                              <div className="mt-2">
                                <p className="font-bold mb-2">
                                  Confusion Matrix:
                                </p>
                                <div className="grid grid-cols-2 gap-2">
                                  {value[0].map((col, colIndex) => (
                                    <div
                                      key={`header-${colIndex}`}
                                      className="w-8 h-8 flex items-center justify-center border bg-gray-100"
                                    >
                                      {col}
                                    </div>
                                  ))}
                                  {value[1].map((col, colIndex) => (
                                    <div
                                      key={`footer-${colIndex}`}
                                      className="w-8 h-8 flex items-center justify-center border bg-gray-100"
                                    >
                                      {col}
                                    </div>
                                  ))}
                                </div>
                              </div>
                            )}
                          </li>
                        ))}
                      </ul>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Tahmin ekranı */}
      {currentScreen === "predict" && (
        <div>
          <h1 className="text-2xl font-bold text-center mb-4">Tahmin Yap</h1>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {featureOptions.map((feature, index) => (
              <select
                key={index}
                value={inputData[index]}
                onChange={(e) => {
                  const newInputData = [...inputData];
                  newInputData[index] = e.target.value;
                  setInputData(newInputData);
                }}
                className="border p-2 rounded w-full"
              >
                <option value="" disabled>
                  {feature.placeholder}
                </option>
                {feature.options.map((option, optIndex) => (
                  <option key={optIndex} value={option}>
                    {option}
                  </option>
                ))}
              </select>
            ))}
          </div>
          <button
            className="mt-4 px-4 py-2 bg-green-500 text-white rounded"
            onClick={handlePrediction}
          >
            Tahmin Yap
          </button>
          {prediction && (
            <div className="mt-4 p-4 border rounded">
              <p className="font-bold">
                Sonuç:{" "}
                <span
                  className={
                    prediction.prediction === 1
                      ? "text-red-500"
                      : "text-green-500"
                  }
                >
                  {prediction.prediction === 1 ? "Zehirli" : "Zehirli Değil"}
                </span>
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
