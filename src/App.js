import React, { useEffect, useState } from "react";

function App() {
  const [currentScreen, setCurrentScreen] = useState("results"); // Ekran durumu
  const [results, setResults] = useState([]); // Model sonuçları
  const [inputData, setInputData] = useState(Array(21).fill("")); // Kullanıcı girdileri
  const [prediction, setPrediction] = useState(null); // Tahmin sonucu

  const labelEncoders = [
    { çıngırak: 0, konik: 1, dışbükey: 2, düz: 3, topuzlu: 4, batık: 5 }, // Özellik 1
    { lifli: 0, oluklar: 1, pullu: 2, düz: 3 }, // Özellik 2
    {
      kahverengi: 0,
      devetüyü: 1,
      tarçın: 2,
      gri: 3,
      yeşil: 4,
      pembe: 5,
      mor: 6,
      kırmızı: 7,
      beyaz: 8,
    }, // Özellik 3
    { var: 0, yok: 1 }, // Özellik 4
    {
      badem: 0,
      anason: 1,
      kreozot: 2,
      şüpheli: 3,
      faul: 4,
      küflü: 5,
      hiçbiri: 6,
      keskin: 7,
      baharatlı: 8,
    }, // Özellik 5
    { ekli: 0, alçalan: 1, özgür: 2, çentikli: 3 }, // Özellik 6
    { kapalı: 0, sık: 1, ayrık: 2 }, // Özellik 7
    { geniş: 0, dar: 1 }, // Özellik 8
    {
      siyah: 0,
      kahverengi: 1,
      devetüyü: 2,
      çikolata: 3,
      gri: 4,
      yeşil: 5,
      turuncu: 6,
      pembe: 7,
      mor: 8,
      kırmızı: 9,
      beyaz: 10,
    }, // Özellik 9
    { genişleme: 0, sivrilen: 1 }, // Özellik 10
    { soğanlı: 0, kulüp: 1, bardak: 2, eşit: 3, rizomorflar: 4, köklü: 5 }, // Özellik 11
    { lifli: 0, pullu: 1, ipeksi: 2, düz: 3 }, // Özellik 12
    { lifli: 0, pullu: 1, ipeksi: 2, düz: 3 }, // Özellik 13
    {
      kahverengi: 0,
      devetüyü: 1,
      tarçın: 2,
      gri: 3,
      turuncu: 4,
      pembe: 5,
      kırmızı: 6,
      beyaz: 7,
      sarı: 8,
    }, // Özellik 14
    {
      kahverengi: 0,
      devetüyü: 1,
      tarçın: 2,
      gri: 3,
      turuncu: 4,
      pembe: 5,
      kırmızı: 6,
      beyaz: 7,
      sarı: 8,
    }, // Özellik 15
    { kahverengi: 0, turuncu: 1, beyaz: 2, sarı: 3 }, // Özellik 16
    { hiçbiri: 0, 1: 1, 2: 2 }, // Özellik 17
    {
      "örümcek ağı": 0,
      geçici: 1,
      "ışıl ışıl": 2,
      büyük: 3,
      hiçbiri: 4,
      kolye: 5,
      mantolama: 6,
      alan: 7,
    }, // Özellik 18
    {
      siyah: 0,
      kahverengi: 1,
      devetüyü: 2,
      çikolata: 3,
      yeşil: 4,
      turuncu: 5,
      mor: 6,
      beyaz: 7,
      sarı: 8,
    }, // Özellik 19
    { bolluk: 0, kümelenmiş: 1, çeşitli: 2, dağınık: 3, birçok: 4, yalnız: 5 }, // Özellik 20
    {
      otlar: 0,
      yapraklar: 1,
      çayırlar: 2,
      yollar: 3,
      kentsel: 4,
      atık: 5,
      orman: 6,
    }, // Özellik 21
  ];

  // Özellikler için seçenekler
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
        "sarı",
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
        "sarı",
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

  // Sonuçları çekme
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

  // Tahmin yapma
  const handlePrediction = () => {
    // String girdileri sayısal değerlere çevirme
    const numericInput = inputData.map(
      (value, index) => labelEncoders[index][value] || 0
    );

    fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ input: numericInput }),
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
        <div className="m-12">
          <h1 className="text-2xl font-bold text-center mb-4">
            Model Sonuçları
          </h1>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {results.map((step, index) => (
              <div
                key={index}
                className="border p-4 rounded-lg shadow-md bg-gray-200"
              >
                <h2 className="text-lg font-bold">{step.step}</h2>
                <div className="mt-2 space-y-4">
                  {step.results.map((result, idx) => (
                    <div
                      key={idx}
                      className="border p-2 rounded shadow-sm bg-white"
                    >
                      <h3 className="font-semibold text-blue-500">
                        {result.model}
                      </h3>
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
        <div className="m-12">
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
