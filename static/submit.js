function getTicker(ticker) {
    axios.post("/endpoint", {clicked_ticker: ticker})
    .then(response => {
        console.log("Clicked Ticker:", response.data);
        window.location.href="/analytics.html";
    })
    .catch(err => console.error(err));
}