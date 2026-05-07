import FinanceDataReader as fdr
import yfinance as yf
import pandas as pd
from tqdm import tqdm

class DataHandler_US:
    def __init__(self, start, end, top_n=300):
        self.start = start
        self.end = end
        self.top_n = top_n

        self.tickers = None
        self.price = None
        self.volume = None
        self.returns = None
        self.mask = None

        self._load_universe()
        self._download_data()
        self._process()

    # 1. Universe (S&P500)
    def _load_universe(self):
        url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
        table = pd.read_csv(url)

        tickers = table['Symbol'].tolist()
        tickers = [t.replace('.', '-') for t in tickers]

        self.tickers = tickers

    # 2. Data download (batch)
    def _download_data(self):

        data = yf.download(
            self.tickers,
            start=self.start,
            end=self.end,
            group_by='ticker',
            auto_adjust=True,
            progress=True
        )

        self.raw = data

    # 3. Process (FDR compatibility)
    def _process(self):

        price_dict = {}
        volume_dict = {}

        for t in self.tickers:
            try:
                df = self.raw[t]

                if df.empty:
                    continue

                price_dict[t] = df['Close']
                volume_dict[t] = df['Volume']

            except:
                continue

        price = pd.DataFrame(price_dict)
        volume = pd.DataFrame(volume_dict)

        price = price.sort_index()
        volume = volume.sort_index()

        # liquidity filter (top N)
        dollar_vol = price * volume
        avg_dv = dollar_vol.rolling(20).mean()

        latest = avg_dv.iloc[-1].dropna()
        top = latest.nlargest(self.top_n).index

        price = price[top]
        volume = volume[top]

        # mask + returns
        self.mask = price.notna()

        price = price.ffill()
        price = price.where(self.mask)

        # returns
        returns = price.pct_change(fill_method=None)
        returns = returns.ffill(limit=3)
        returns = returns.where(self.mask)

        self.price = price
        self.volume = volume
        self.returns = returns

    def get_price(self):
        return self.price

    def get_returns(self):
        return self.returns

    def get_volume(self):
        return self.volume

    def get_mask(self):
        return self.mask

class DataHandler_KR:
    def __init__(self, start, end, universe="ALL", top_liq=300):
        self.start = start
        self.end = end
        self.universe = universe
        self.top_liq = top_liq

        self.tickers = None
        self.price = None
        self.volume = None
        self.returns = None
        self.mask = None

        self._load_universe()
        self._download_data()
        self._process()

    def _load_universe(self):
        kospi = fdr.StockListing('KOSPI')
        kosdaq = fdr.StockListing('KOSDAQ')

        # Market Selection
        if self.universe == "KOSPI":
            self.tickers = kospi['Code'].tolist()
        elif self.universe == "KOSDAQ":
            self.tickers = kosdaq['Code'].tolist()
        else:
            self.tickers = pd.concat([kospi, kosdaq])['Code'].tolist()

    def _download_data(self):
        data = []

        for t in tqdm(self.tickers):
            try:
                df = fdr.DataReader(t, self.start, self.end)[['Close', 'Volume']]
                df['ticker'] = t
                data.append(df)
            except:
                continue

        self.raw = pd.concat(data)

    def _process(self):
        df = self.raw.reset_index()

        price = df.pivot(index='Date', columns='ticker', values='Close').sort_index()
        volume = df.pivot(index='Date', columns='ticker', values='Volume').sort_index()

        # 1. mask
        mask = price.notna()

        # 2. fill
        price = price.ffill()
        volume = volume.ffill()

        # 3. mask
        price = price.where(mask)

        # 4. liquidity filter
        dollar_vol = price * volume
        avg_dv = dollar_vol.rolling(20).mean()

        latest = avg_dv.iloc[-1].dropna()
        top = latest.nlargest(self.top_liq).index

        price = price[top]
        volume = volume[top]
        mask = mask[top]

        # 5. returns
        returns = price.pct_change(fill_method=None)
        returns = returns.ffill(limit=3)
        returns = returns.where(mask)

        self.price = price
        self.volume = volume
        self.returns = returns
        self.mask = mask

    def get_price(self):
        return self.price

    def get_returns(self):
        return self.returns

    def get_mask(self):
        return self.mask

    def get_volume(self):
        return self.volume