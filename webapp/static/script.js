document.addEventListener("DOMContentLoaded", function () {
  const form = document.getElementById("analyze-form");
  const articleTextEl = document.getElementById("article_text");
  const errorEl = document.getElementById("error-message");
  const resultsSection = document.getElementById("results-section");

  const predictedLabelEl = document.getElementById("predicted-label");
  const fakeProbEl = document.getElementById("fake-prob");
  const realProbEl = document.getElementById("real-prob");
  const clusterNameEl = document.getElementById("cluster-name");
  const clusterSizeEl = document.getElementById("cluster-size");
  const clusterTermsEl = document.getElementById("cluster-terms");

  const analyzeBtn = document.getElementById("analyze-btn");

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    errorEl.classList.add("hidden");
    errorEl.textContent = "";
    resultsSection.classList.add("hidden");

    const text = articleTextEl.value.trim();
    if (!text) {
      errorEl.textContent = "Please paste some article text first.";
      errorEl.classList.remove("hidden");
      return;
    }

    analyzeBtn.disabled = true;
    analyzeBtn.textContent = "Analyzing...";

    const formData = new FormData();
    formData.append("article_text", text);

    fetch("/analyze", {
      method: "POST",
      body: formData
    })
      .then((response) => {
        if (!response.ok) {
          return response.json().then((data) => {
            throw new Error(data.error || "Server error");
          });
        }
        return response.json();
      })
      .then((data) => {
        // Classification display
        const label = data.label;
        const fakeProb = data.fake_probability;
        const realProb = data.real_probability;

        predictedLabelEl.textContent = label === "FAKE"
          ? `This article is likely FAKE`
          : `This article is likely REAL`;

        predictedLabelEl.classList.remove("fake", "real");
        if (label === "FAKE") {
          predictedLabelEl.classList.add("fake");
        } else {
          predictedLabelEl.classList.add("real");
        }

        fakeProbEl.textContent = `Fake probability: ${(fakeProb * 100).toFixed(1)}%`;
        realProbEl.textContent = `Real probability: ${(realProb * 100).toFixed(1)}%`;

        clusterNameEl.textContent = data.cluster_name || `Cluster ${data.cluster_id}`;
        clusterSizeEl.textContent = data.cluster_size || 0;

        clusterTermsEl.innerHTML = "";
        if (Array.isArray(data.cluster_top_terms)) {
          data.cluster_top_terms.slice(0, 10).forEach((term) => {
            const li = document.createElement("li");
            li.textContent = term;
            clusterTermsEl.appendChild(li);
          });
        }

        resultsSection.classList.remove("hidden");
      })
      .catch((err) => {
        console.error(err);
        errorEl.textContent = err.message || "An error occurred while analyzing the article.";
        errorEl.classList.remove("hidden");
      })
      .finally(() => {
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = "Analyze";
      });
  });
});
