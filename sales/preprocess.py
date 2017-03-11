import numpy as np
import pandas as pd
from scipy import stats

class preprocess_data(object):
    def __init__(self):
        self.f = open("sales_data.txt")
        self.text = open('temperature.txt')

    def order_data(self):
        f = self.f
        sales = []
        count = 0
        for line in f.readlines():
            if line[-1] == '\n':
                sales.append(line[:-1])
            count = count + 1
        data = []
        count_min = 0
        count_max = int((count - 4) / 2)

        for i in range(count_min, count_max):
            data.append(sales[2 * i + 4])

        data = np.asarray(data)
        Rdata = []
        date = []
        sales_record = []

        for i in range(len(data) - 2):
            Rdata.append(data[i][1:-1])
            date.append(Rdata[i][:10])
            Sdata = Rdata[i][-10:].strip()
            sales_record.append(Sdata.replace(',', ''))

        date = np.asarray(date)
        date = pd.to_datetime(date)
        sales_record = np.asarray(sales_record)
        return date, sales_record


    def make_dataframe(self, date, sales):
        dic = {'date': date, 'sales': sales, 'weekday': date.weekday}
        df = pd.DataFrame(data=dic, columns=['sales', 'weekday'])
        df.index = pd.Index(date)
        return df


    def cut_data(self, dataframe):
        dataframe = dataframe.sort(['sales'], ascending=[0])
        num = int(len(dataframe) * 0.2)
        dataframe_high = dataframe[:num]
        dataframe_low = dataframe[-num:]
        return dataframe_high, dataframe_low


    def divide_set(self,df):
        vacation = []
        semester = []
        for period in ['2013', '2014', '2015', '2016']:
            for i in range(1, 13):
                string = '-' + str(i)
                if string == '-1' or string == '-2' or string == '-7' or string == '-8':
                    try:
                        vacation.append(df[period + string])
                    except:
                        continue
                else:
                    try:
                        semester.append(df[period + string])
                    except:
                        continue
        vacation = pd.concat(vacation)
        semester = pd.concat(semester)
        return vacation, semester


    def cut_data(self, dataframe):
        dataframe = dataframe.sort_values(by='sales', ascending=[0])
        num = int(len(dataframe) * 0.2)
        dataframe_high = dataframe[:num]
        dataframe_low = dataframe[-num:]
        return dataframe_high, dataframe_low

    def load_weather(self,text):
        text = self.text
        res = []

        for line in text.readlines():
            line = line.strip().split('\t')
            res.append(line)

        num = int(len(res) / 4)
        fst = res[:num]
        snd = res[num:2 * num]
        thd = res[2 * num:3 * num]
        fth = res[3 * num:]

        return fst, snd, thd, fth

    def col_data(self, data, num):
        res = []
        try:
            for j in data[1:]:
                res.append(j[num])
        except:
            pass
        return res


    def merge_data(self, data):
        month = data[0]
        res = []

        for i in range(1, len(month) + 1):
            res.extend(self.col_data(data, i))

        return res


    def weather_df(self, weather, date):
        start = '2013-3-22'
        end = '2016-3-22'
        period = pd.date_range(start, end, freq='D')

        '''
         * 0 : Mon ~ 6 : Sun
        '''
        dic = {'date': period, 'weather': weather}
        df = pd.DataFrame(data=dic, columns=['weather'])
        df.index = pd.Index(period)
        df = df.drop(period.difference(date))

        return df

    def classify_sales(self, data):
        data = [int(a[0:-1]) for a in data]
        value = []

        for sale in data:
            if int(sale) < 1.117600e+06:
                value.append(0)
            elif int(sale) >= 1.117600e+06 and int(sale) < 1.362900e+06:
                value.append(1)
            elif int(sale) >= 1.362900e+06 and int(sale) < 1.700400e+06:
                value.append(2)
            elif int(sale) >= 1.700400e+06:
                value.append(3)

        return value

    def get_data(self):
        date, sales = self.order_data()
        sales = self.classify_sales(sales)
        df = self.make_dataframe(date, sales)
        fst, snd, thd, fth = self.load_weather(self.text)

        weather = []
        for data in [fst, snd, thd, fth]:
            tmp = self.merge_data(data)
            weather.extend(tmp)
        weather = [float(a) for a in weather if not a == '' and not a == ' ']
        weatherDF = self.weather_df(weather, date)
        df['temp'] = weatherDF.weather

        '''
         * vacation : 1 , semester : 0
        '''
        vacation, semester = self.divide_set(df)
        vacation['vacation'] = 1
        semester['vacation'] = 0

        df = pd.concat([vacation, semester])
        df = df.sort_index()

        data = df.copy()
        data = data.dropna()

        return data
