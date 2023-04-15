from utils import *
from models import *

class QualitativePredictor:
    def __init__(self, stock):
        self.stock = stock
        self.data = self.process_data(stock)
        self.model = RiskPredictor(in_channels=1, out_channels=64, projected_dim=128, hidden_dim=64, num_layers=2, num_heads=2, kernel_size=5, use_bias=False)

    def fit(self, max_epochs=20, verbose=True):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        self.model.train()
        for epoch in range(1, max_epochs+1):
            all_loss = []
            for (x, t, y) in self.data:
                optimizer.zero_grad()
                out = self.model(x, t)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                all_loss.append(loss.item())
            if epoch % (max_epochs//5) == 0 and verbose:
                print(f"Epoch: {epoch} | Loss: {np.array(all_loss).mean()}")

    def predict_next(self):
        self.model.eval()
        with torch.no_grad():
            return self.model(self.data[-1][0], self.data[-1][1])
    
    @staticmethod
    def process_data(stock):
        stock_file = os.path.join("../FMP/", stock, "price.csv")

        stock_data = pd.read_csv(stock_file, parse_dates=['ds'])
        stock_data['log_returns'] = np.log(stock_data['close'] / stock_data['close'].shift(1))
        stock_data.dropna(inplace=True)
        stock_data = stock_data

        f = open(os.path.join("../FMP/", stock, "2022", "2022_1.json"), "r")
        sentiment_q1 = json.load(f)
        f.close()

        f = open(os.path.join("../FMP/", stock, "2022", "2022_2.json"), "r")
        sentiment_q2 = json.load(f)
        f.close()

        f = open(os.path.join("../FMP/", stock, "2022", "2022_3.json"), "r")
        sentiment_q3 = json.load(f)
        f.close()

        f = open(os.path.join("../FMP/", stock, "2022", "2022_4.json"), "r")
        sentiment_q4 = json.load(f)
        f.close()

        f = open(os.path.join("../FMP/", stock, "2023", "2023_1.json"), "r")
        sentiment_q5 = json.load(f)
        f.close()

        stock_data_q1 = stock_data[(stock_data['ds']>='2021-01-01') & (stock_data['ds']<sentiment_q1['date'])]
        stock_data_q1_embed = torch.tensor(get_embedding(" ".join(sentiment_q1['content'].split(" ")[:6400]))).unsqueeze(0)

        stock_data_q2 = stock_data[(stock_data['ds']>=sentiment_q1['date']) & (stock_data['ds']<sentiment_q2['date'])]
        stock_data_q2_embed = torch.tensor(get_embedding(" ".join(sentiment_q2['content'].split(" ")[:6400]))).unsqueeze(0)

        stock_data_q3 = stock_data[(stock_data['ds']>=sentiment_q2['date']) & (stock_data['ds']<sentiment_q3['date'])]
        stock_data_q3_embed = torch.tensor(get_embedding(" ".join(sentiment_q3['content'].split(" ")[:6400]))).unsqueeze(0)

        stock_data_q4 = stock_data[(stock_data['ds']>=sentiment_q3['date']) & (stock_data['ds']<sentiment_q4['date'])]
        stock_data_q4_embed = torch.tensor(get_embedding(" ".join(sentiment_q4['content'].split(" ")[:6400]))).unsqueeze(0)

        stock_data_q5 = stock_data[(stock_data['ds']>=sentiment_q4['date']) & (stock_data['ds']<sentiment_q5['date'])]

        data = []
        data.append(QualitativePredictor.prepare_data(stock_data_q1, stock_data_q1_embed, stock_data_q2))
        data.append(QualitativePredictor.prepare_data(stock_data_q2, stock_data_q2_embed, stock_data_q3))
        data.append(QualitativePredictor.prepare_data(stock_data_q3, stock_data_q3_embed, stock_data_q4))
        data.append(QualitativePredictor.prepare_data(stock_data_q4, stock_data_q4_embed, stock_data_q5))
        return data

    @staticmethod
    def prepare_data(stock, embedding, projected):
        prices = torch.tensor(stock['close'].values).unsqueeze(0).unsqueeze(0)
        risk = torch.tensor(compute_historic_volatility(projected, -1).values).unsqueeze(0)
        return prices.float(), embedding.float(), risk.float()
    

if  __name__=="__main__":
    predictor = QualitativePredictor("AAPL")
    predictor.fit(max_epochs=20)

    print(predictor.predict_next())