{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Multiple Linear Regression\n",
    "- Kali ini Ucup mencoba untuk membantu temenya yaitu Joko untuk memprediksi harga rumah untuknya.\n",
    "- Joko senduri tinggal di US tepatnya di King County dan sekarang sedang mencari rumah karena dia baru saja menikaj.\n",
    "- Data diambil dari kaggle dengan sedikit modifikasi.\n",
    "- Joko sendiri ingin membeli rumah dengan jumlah kamar tidur itu 3, jumlah kamar mandinya itu 2, luas rumahnya itu 1800sqft, grade rumahnya 7 dan tahun pembangunanya pada tahun 1990.\n",
    "- Yuk bantu Ucup membangun model machine learning untuk membantu joko!"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- Langkah Pengerjaan hampir sama dengan yang Simple Linear Regression hanya saja Multivariate Linear Regression memiliki lebih > 1 independent variable (x)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Load library"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.model_selection import train_test_split"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Load datasets"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Nama dataframe kita adalah df yang berisi data dari kc_house_data.csv.\n",
    "#Features yang digunakan adalah 'bedrooms','bathrooms','sqft_living','grade','price' dan 'yr_built'\n",
    "df = pd.read_csv('kc_house_data.csv', usecols=['bedrooms', 'bathrooms', 'sqft_living', 'grade', 'price', 'yr_built'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Sneak peak data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>price</th>\n",
       "      <th>bedrooms</th>\n",
       "      <th>bathrooms</th>\n",
       "      <th>sqft_living</th>\n",
       "      <th>grade</th>\n",
       "      <th>yr_built</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>221900.0</td>\n",
       "      <td>3</td>\n",
       "      <td>1.00</td>\n",
       "      <td>1180</td>\n",
       "      <td>7</td>\n",
       "      <td>1955</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>538000.0</td>\n",
       "      <td>3</td>\n",
       "      <td>2.25</td>\n",
       "      <td>2570</td>\n",
       "      <td>7</td>\n",
       "      <td>1951</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>180000.0</td>\n",
       "      <td>2</td>\n",
       "      <td>1.00</td>\n",
       "      <td>770</td>\n",
       "      <td>6</td>\n",
       "      <td>1933</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>604000.0</td>\n",
       "      <td>4</td>\n",
       "      <td>3.00</td>\n",
       "      <td>1960</td>\n",
       "      <td>7</td>\n",
       "      <td>1965</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>510000.0</td>\n",
       "      <td>3</td>\n",
       "      <td>2.00</td>\n",
       "      <td>1680</td>\n",
       "      <td>8</td>\n",
       "      <td>1987</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      price  bedrooms  bathrooms  sqft_living  grade  yr_built\n",
       "0  221900.0         3       1.00         1180      7      1955\n",
       "1  538000.0         3       2.25         2570      7      1951\n",
       "2  180000.0         2       1.00          770      6      1933\n",
       "3  604000.0         4       3.00         1960      7      1965\n",
       "4  510000.0         3       2.00         1680      8      1987"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Melihat 5 baris teratas dari data\n",
    "#Independent variabel(x) adalah bedrooms, bathrooms, sqft_living, grade, yr_built\n",
    "#Dependent variabel(y) adalah price\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- Penjelasan setiap kolom:\n",
    "    1. bedrooms = Jumlah kamar tidur\n",
    "    2. bathrooms = Jumlah kamar mandi\n",
    "    3. sqft_living = Luas rumah dalam satuan sqft\n",
    "    4. grade = Grading system dari pemerintah King County US\n",
    "    5. yr_built = Tahun dimana rumah dibangun\n",
    "    6. price = Harga dari rumah (US$)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(21613, 6)"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Mengetahui jumlah kolom dan baris dari data\n",
    "#Data kita mempunya 6 kolom (features) dengan 21613 baris\n",
    "df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 21613 entries, 0 to 21612\n",
      "Data columns (total 6 columns):\n",
      "price          21613 non-null float64\n",
      "bedrooms       21613 non-null int64\n",
      "bathrooms      21613 non-null float64\n",
      "sqft_living    21613 non-null int64\n",
      "grade          21613 non-null int64\n",
      "yr_built       21613 non-null int64\n",
      "dtypes: float64(2), int64(4)\n",
      "memory usage: 1013.2 KB\n"
     ]
    }
   ],
   "source": [
    "#Melihat informasi data kita mulai dari jumlah data, tipe data, memory yang digunakan dll.\n",
    "#Dapat dilihat bahwa seluruh data sudah di dalam bentuk numerik\n",
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>price</th>\n",
       "      <th>bedrooms</th>\n",
       "      <th>bathrooms</th>\n",
       "      <th>sqft_living</th>\n",
       "      <th>grade</th>\n",
       "      <th>yr_built</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>2.161300e+04</td>\n",
       "      <td>21613.000000</td>\n",
       "      <td>21613.000000</td>\n",
       "      <td>21613.000000</td>\n",
       "      <td>21613.000000</td>\n",
       "      <td>21613.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>5.400881e+05</td>\n",
       "      <td>3.370842</td>\n",
       "      <td>2.114757</td>\n",
       "      <td>2079.899736</td>\n",
       "      <td>7.656873</td>\n",
       "      <td>1971.005136</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>3.671272e+05</td>\n",
       "      <td>0.930062</td>\n",
       "      <td>0.770163</td>\n",
       "      <td>918.440897</td>\n",
       "      <td>1.175459</td>\n",
       "      <td>29.373411</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>7.500000e+04</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>290.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1900.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>3.219500e+05</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>1.750000</td>\n",
       "      <td>1427.000000</td>\n",
       "      <td>7.000000</td>\n",
       "      <td>1951.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>4.500000e+05</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>2.250000</td>\n",
       "      <td>1910.000000</td>\n",
       "      <td>7.000000</td>\n",
       "      <td>1975.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>6.450000e+05</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>2.500000</td>\n",
       "      <td>2550.000000</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>1997.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>7.700000e+06</td>\n",
       "      <td>33.000000</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>13540.000000</td>\n",
       "      <td>13.000000</td>\n",
       "      <td>2015.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              price      bedrooms     bathrooms   sqft_living         grade  \\\n",
       "count  2.161300e+04  21613.000000  21613.000000  21613.000000  21613.000000   \n",
       "mean   5.400881e+05      3.370842      2.114757   2079.899736      7.656873   \n",
       "std    3.671272e+05      0.930062      0.770163    918.440897      1.175459   \n",
       "min    7.500000e+04      0.000000      0.000000    290.000000      1.000000   \n",
       "25%    3.219500e+05      3.000000      1.750000   1427.000000      7.000000   \n",
       "50%    4.500000e+05      3.000000      2.250000   1910.000000      7.000000   \n",
       "75%    6.450000e+05      4.000000      2.500000   2550.000000      8.000000   \n",
       "max    7.700000e+06     33.000000      8.000000  13540.000000     13.000000   \n",
       "\n",
       "           yr_built  \n",
       "count  21613.000000  \n",
       "mean    1971.005136  \n",
       "std       29.373411  \n",
       "min     1900.000000  \n",
       "25%     1951.000000  \n",
       "50%     1975.000000  \n",
       "75%     1997.000000  \n",
       "max     2015.000000  "
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Melihat statistical description dari data mulai dari mean, kuartil, standard deviation dll\n",
    "df.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- Pada feature bathrooms terdapat nilai pecahan, aneh kan yak kalo ada nilai jumlah kamar mandi pecahan gitu. Maka kita ubah dulu jenis data yang semula float menjadi int.\n",
    "- Pada feature bedrooms terdapat nilai 33, ini sangat aneh karena masak rumah ada yang punya kamar 33 apalagi ini rumah pribadi. jadi kemungkinan itu typo dan akan saya ganti menjadi 3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Mrubah tipe data dari bathrooms yang semula float menjadi int\n",
    "df['bathrooms'] = df['bathrooms'].astype('int')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Mengganti nilai 33 menjadi 3\n",
    "df['bedrooms'] = df['bedrooms'].replace(33,3)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Handling Missing Values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "price          0\n",
       "bedrooms       0\n",
       "bathrooms      0\n",
       "sqft_living    0\n",
       "grade          0\n",
       "yr_built       0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Mencari dan menangani missing values\n",
    "#Ternyata data kita tidak ada missing values\n",
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Exploratory Data Analysis (EDA)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>price</th>\n",
       "      <th>bedrooms</th>\n",
       "      <th>bathrooms</th>\n",
       "      <th>sqft_living</th>\n",
       "      <th>grade</th>\n",
       "      <th>yr_built</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>221900.0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>1180</td>\n",
       "      <td>7</td>\n",
       "      <td>1955</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>538000.0</td>\n",
       "      <td>3</td>\n",
       "      <td>2</td>\n",
       "      <td>2570</td>\n",
       "      <td>7</td>\n",
       "      <td>1951</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>180000.0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>770</td>\n",
       "      <td>6</td>\n",
       "      <td>1933</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>604000.0</td>\n",
       "      <td>4</td>\n",
       "      <td>3</td>\n",
       "      <td>1960</td>\n",
       "      <td>7</td>\n",
       "      <td>1965</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>510000.0</td>\n",
       "      <td>3</td>\n",
       "      <td>2</td>\n",
       "      <td>1680</td>\n",
       "      <td>8</td>\n",
       "      <td>1987</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      price  bedrooms  bathrooms  sqft_living  grade  yr_built\n",
       "0  221900.0         3          1         1180      7      1955\n",
       "1  538000.0         3          2         2570      7      1951\n",
       "2  180000.0         2          1          770      6      1933\n",
       "3  604000.0         4          3         1960      7      1965\n",
       "4  510000.0         3          2         1680      8      1987"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 864x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Univariate analysis bedrooms\n",
    "#Melihat distribusi dari bedrooms\n",
    "f = plt.figure(figsize=(12,4))\n",
    "\n",
    "f.add_subplot(1,2,1)\n",
    "sns.countplot(df['bedrooms'])\n",
    "\n",
    "f.add_subplot(1,2,2)\n",
    "plt.boxplot(df['bedrooms'])\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- Dapat dilihat bahwa sebagian besar jumlah kamar tidur itu di angka 3 dan 4.\n",
    "- Data memiliki banyak outliers."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 864x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Univariate analysis bathrooms\n",
    "#Melihat distribusi dari bathrooms\n",
    "f = plt.figure(figsize=(12,4))\n",
    "\n",
    "f.add_subplot(1,2,1)\n",
    "sns.countplot(df['bathrooms'])\n",
    "\n",
    "f.add_subplot(1,2,2)\n",
    "plt.boxplot(df['bathrooms'])\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- Jumlah kamar mandi paling banyak berada pada angka 1 dan 2.\n",
    "- Yang menarik disini adalah dimana ada rumah yang tidak ada kamar mandinya atau jumlahnya 0\n",
    "- Nilai outlier sendiri lumayan banyak."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 864x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Univariate analysis sqft_living\n",
    "#Melihat distribusi dari sqft_living\n",
    "f = plt.figure(figsize=(12,4))\n",
    "\n",
    "f.add_subplot(1,2,1)\n",
    "df['sqft_living'].plot(kind='kde')\n",
    "\n",
    "f.add_subplot(1,2,2)\n",
    "plt.boxplot(df['sqft_living'])\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- Density dari distribusi luas rumah berada di sekitar angka 2000an.\n",
    "- Banyak terdapat outliers."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 864x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Univariate analysis grade\n",
    "#Melihat distribusi dari grade\n",
    "f = plt.figure(figsize=(12,4))\n",
    "\n",
    "f.add_subplot(1,2,1)\n",
    "sns.countplot(df['grade'])\n",
    "\n",
    "f.add_subplot(1,2,2)\n",
    "plt.boxplot(df['grade'])\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- Sebagian besar rumah di County King US memiliki grade 7 dan 8.\n",
    "- Dilihat dari boxplot, data memiliki beberapa outliers."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1440x576 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Univariate analysis yr_built\n",
    "#Melihat distribusi dari yr_built\n",
    "f = plt.figure(figsize=(20,8))\n",
    "\n",
    "f.add_subplot(1,2,1)\n",
    "sns.countplot(df['yr_built'])\n",
    "\n",
    "f.add_subplot(1,2,2)\n",
    "plt.boxplot(df['yr_built'])\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- Dapat dilihat bahwa semakin tua umur dari rumah, maka semakin sedikit orang yang menjual rumahnya tersebut.\n",
    "- Density terdapat di sekitar tahun 1980an.\n",
    "- Data tidak memiliki outliers."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "F:\\Anaconda\\envs\\pandas\\lib\\site-packages\\seaborn\\axisgrid.py:2065: UserWarning: The `size` parameter has been renamed to `height`; pleaes update your code.\n",
      "  warnings.warn(msg, UserWarning)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 720x576 with 0 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1350x360 with 5 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Bivariate analysis antara independent variable dan dependent variable\n",
    "#Melihat hubungan antara independent dan dependent\n",
    "#Menggunakan pairplot\n",
    "plt.figure(figsize=(10,8))\n",
    "sns.pairplot(data=df, x_vars=['bedrooms', 'bathrooms', 'sqft_living', 'grade', 'yr_built'], y_vars=['price'], size=5, aspect=0.75)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style  type=\"text/css\" >\n",
       "    #T_95779fa8_d6da_11e9_9184_309c23131503row0_col0 {\n",
       "            background-color:  #023858;\n",
       "            color:  #f1f1f1;\n",
       "        }    #T_95779fa8_d6da_11e9_9184_309c23131503row0_col1 {\n",
       "            background-color:  #dfddec;\n",
       "            color:  #000000;\n",
       "        }    #T_95779fa8_d6da_11e9_9184_309c23131503row0_col2 {\n",
       "            background-color:  #eae6f1;\n",
       "            color:  #000000;\n",
       "        }    #T_95779fa8_d6da_11e9_9184_309c23131503row0_col3 {\n",
       "            background-color:  #549cc7;\n",
       "            color:  #000000;\n",
       "        }    #T_95779fa8_d6da_11e9_9184_309c23131503row0_col4 {\n",
       "            background-color:  #7eadd1;\n",
       "            color:  #000000;\n",
       "        }    #T_95779fa8_d6da_11e9_9184_309c23131503row0_col5 {\n",
       "            background-color:  #fff7fb;\n",
       "            color:  #000000;\n",
       "        }    #T_95779fa8_d6da_11e9_9184_309c23131503row1_col0 {\n",
       "            background-color:  #c8cde4;\n",
       "            color:  #000000;\n",
       "        }    #T_95779fa8_d6da_11e9_9184_309c23131503row1_col1 {\n",
       "            background-color:  #023858;\n",
       "            color:  #f1f1f1;\n",
       "        }    #T_95779fa8_d6da_11e9_9184_309c23131503row1_col2 {\n",
       "            background-color:  #f2ecf5;\n",
       "            color:  #000000;\n",
       "        }    #T_95779fa8_d6da_11e9_9184_309c23131503row1_col3 {\n",
       "            background-color:  #9cb9d9;\n",
       "            color:  #000000;\n",
       "        }    #T_95779fa8_d6da_11e9_9184_309c23131503row1_col4 {\n",
       "            background-color:  #fff7fb;\n",
       "            color:  #000000;\n",
       "        }    #T_95779fa8_d6da_11e9_9184_309c23131503row1_col5 {\n",
       "            background-color:  #eee9f3;\n",
       "            color:  #000000;\n",
       "        }    #T_95779fa8_d6da_11e9_9184_309c23131503row2_col0 {\n",
       "            background-color:  #7bacd1;\n",
       "            color:  #000000;\n",
       "        }    #T_95779fa8_d6da_11e9_9184_309c23131503row2_col1 {\n",
       "            background-color:  #a4bcda;\n",
       "            color:  #000000;\n",
       "        }    #T_95779fa8_d6da_11e9_9184_309c23131503row2_col2 {\n",
       "            background-color:  #023858;\n",
       "            color:  #f1f1f1;\n",
       "        }    #T_95779fa8_d6da_11e9_9184_309c23131503row2_col3 {\n",
       "            background-color:  #589ec8;\n",
       "            color:  #000000;\n",
       "        }    #T_95779fa8_d6da_11e9_9184_309c23131503row2_col4 {\n",
       "            background-color:  #a4bcda;\n",
       "            color:  #000000;\n",
       "        }    #T_95779fa8_d6da_11e9_9184_309c23131503row2_col5 {\n",
       "            background-color:  #9cb9d9;\n",
       "            color:  #000000;\n",
       "        }    #T_95779fa8_d6da_11e9_9184_309c23131503row3_col0 {\n",
       "            background-color:  #1e80b8;\n",
       "            color:  #000000;\n",
       "        }    #T_95779fa8_d6da_11e9_9184_309c23131503row3_col1 {\n",
       "            background-color:  #6da6cd;\n",
       "            color:  #000000;\n",
       "        }    #T_95779fa8_d6da_11e9_9184_309c23131503row3_col2 {\n",
       "            background-color:  #81aed2;\n",
       "            color:  #000000;\n",
       "        }    #T_95779fa8_d6da_11e9_9184_309c23131503row3_col3 {\n",
       "            background-color:  #023858;\n",
       "            color:  #f1f1f1;\n",
       "        }    #T_95779fa8_d6da_11e9_9184_309c23131503row3_col4 {\n",
       "            background-color:  #358fc0;\n",
       "            color:  #000000;\n",
       "        }    #T_95779fa8_d6da_11e9_9184_309c23131503row3_col5 {\n",
       "            background-color:  #c6cce3;\n",
       "            color:  #000000;\n",
       "        }    #T_95779fa8_d6da_11e9_9184_309c23131503row4_col0 {\n",
       "            background-color:  #2c89bd;\n",
       "            color:  #000000;\n",
       "        }    #T_95779fa8_d6da_11e9_9184_309c23131503row4_col1 {\n",
       "            background-color:  #d1d2e6;\n",
       "            color:  #000000;\n",
       "        }    #T_95779fa8_d6da_11e9_9184_309c23131503row4_col2 {\n",
       "            background-color:  #bdc8e1;\n",
       "            color:  #000000;\n",
       "        }    #T_95779fa8_d6da_11e9_9184_309c23131503row4_col3 {\n",
       "            background-color:  #2c89bd;\n",
       "            color:  #000000;\n",
       "        }    #T_95779fa8_d6da_11e9_9184_309c23131503row4_col4 {\n",
       "            background-color:  #023858;\n",
       "            color:  #f1f1f1;\n",
       "        }    #T_95779fa8_d6da_11e9_9184_309c23131503row4_col5 {\n",
       "            background-color:  #96b6d7;\n",
       "            color:  #000000;\n",
       "        }    #T_95779fa8_d6da_11e9_9184_309c23131503row5_col0 {\n",
       "            background-color:  #fff7fb;\n",
       "            color:  #000000;\n",
       "        }    #T_95779fa8_d6da_11e9_9184_309c23131503row5_col1 {\n",
       "            background-color:  #fff7fb;\n",
       "            color:  #000000;\n",
       "        }    #T_95779fa8_d6da_11e9_9184_309c23131503row5_col2 {\n",
       "            background-color:  #fff7fb;\n",
       "            color:  #000000;\n",
       "        }    #T_95779fa8_d6da_11e9_9184_309c23131503row5_col3 {\n",
       "            background-color:  #fff7fb;\n",
       "            color:  #000000;\n",
       "        }    #T_95779fa8_d6da_11e9_9184_309c23131503row5_col4 {\n",
       "            background-color:  #ece7f2;\n",
       "            color:  #000000;\n",
       "        }    #T_95779fa8_d6da_11e9_9184_309c23131503row5_col5 {\n",
       "            background-color:  #023858;\n",
       "            color:  #f1f1f1;\n",
       "        }</style><table id=\"T_95779fa8_d6da_11e9_9184_309c23131503\" ><thead>    <tr>        <th class=\"blank level0\" ></th>        <th class=\"col_heading level0 col0\" >price</th>        <th class=\"col_heading level0 col1\" >bedrooms</th>        <th class=\"col_heading level0 col2\" >bathrooms</th>        <th class=\"col_heading level0 col3\" >sqft_living</th>        <th class=\"col_heading level0 col4\" >grade</th>        <th class=\"col_heading level0 col5\" >yr_built</th>    </tr></thead><tbody>\n",
       "                <tr>\n",
       "                        <th id=\"T_95779fa8_d6da_11e9_9184_309c23131503level0_row0\" class=\"row_heading level0 row0\" >price</th>\n",
       "                        <td id=\"T_95779fa8_d6da_11e9_9184_309c23131503row0_col0\" class=\"data row0 col0\" >1</td>\n",
       "                        <td id=\"T_95779fa8_d6da_11e9_9184_309c23131503row0_col1\" class=\"data row0 col1\" >0.32</td>\n",
       "                        <td id=\"T_95779fa8_d6da_11e9_9184_309c23131503row0_col2\" class=\"data row0 col2\" >0.51</td>\n",
       "                        <td id=\"T_95779fa8_d6da_11e9_9184_309c23131503row0_col3\" class=\"data row0 col3\" >0.7</td>\n",
       "                        <td id=\"T_95779fa8_d6da_11e9_9184_309c23131503row0_col4\" class=\"data row0 col4\" >0.67</td>\n",
       "                        <td id=\"T_95779fa8_d6da_11e9_9184_309c23131503row0_col5\" class=\"data row0 col5\" >0.054</td>\n",
       "            </tr>\n",
       "            <tr>\n",
       "                        <th id=\"T_95779fa8_d6da_11e9_9184_309c23131503level0_row1\" class=\"row_heading level0 row1\" >bedrooms</th>\n",
       "                        <td id=\"T_95779fa8_d6da_11e9_9184_309c23131503row1_col0\" class=\"data row1 col0\" >0.32</td>\n",
       "                        <td id=\"T_95779fa8_d6da_11e9_9184_309c23131503row1_col1\" class=\"data row1 col1\" >1</td>\n",
       "                        <td id=\"T_95779fa8_d6da_11e9_9184_309c23131503row1_col2\" class=\"data row1 col2\" >0.48</td>\n",
       "                        <td id=\"T_95779fa8_d6da_11e9_9184_309c23131503row1_col3\" class=\"data row1 col3\" >0.59</td>\n",
       "                        <td id=\"T_95779fa8_d6da_11e9_9184_309c23131503row1_col4\" class=\"data row1 col4\" >0.37</td>\n",
       "                        <td id=\"T_95779fa8_d6da_11e9_9184_309c23131503row1_col5\" class=\"data row1 col5\" >0.16</td>\n",
       "            </tr>\n",
       "            <tr>\n",
       "                        <th id=\"T_95779fa8_d6da_11e9_9184_309c23131503level0_row2\" class=\"row_heading level0 row2\" >bathrooms</th>\n",
       "                        <td id=\"T_95779fa8_d6da_11e9_9184_309c23131503row2_col0\" class=\"data row2 col0\" >0.51</td>\n",
       "                        <td id=\"T_95779fa8_d6da_11e9_9184_309c23131503row2_col1\" class=\"data row2 col1\" >0.48</td>\n",
       "                        <td id=\"T_95779fa8_d6da_11e9_9184_309c23131503row2_col2\" class=\"data row2 col2\" >1</td>\n",
       "                        <td id=\"T_95779fa8_d6da_11e9_9184_309c23131503row2_col3\" class=\"data row2 col3\" >0.7</td>\n",
       "                        <td id=\"T_95779fa8_d6da_11e9_9184_309c23131503row2_col4\" class=\"data row2 col4\" >0.61</td>\n",
       "                        <td id=\"T_95779fa8_d6da_11e9_9184_309c23131503row2_col5\" class=\"data row2 col5\" >0.43</td>\n",
       "            </tr>\n",
       "            <tr>\n",
       "                        <th id=\"T_95779fa8_d6da_11e9_9184_309c23131503level0_row3\" class=\"row_heading level0 row3\" >sqft_living</th>\n",
       "                        <td id=\"T_95779fa8_d6da_11e9_9184_309c23131503row3_col0\" class=\"data row3 col0\" >0.7</td>\n",
       "                        <td id=\"T_95779fa8_d6da_11e9_9184_309c23131503row3_col1\" class=\"data row3 col1\" >0.59</td>\n",
       "                        <td id=\"T_95779fa8_d6da_11e9_9184_309c23131503row3_col2\" class=\"data row3 col2\" >0.7</td>\n",
       "                        <td id=\"T_95779fa8_d6da_11e9_9184_309c23131503row3_col3\" class=\"data row3 col3\" >1</td>\n",
       "                        <td id=\"T_95779fa8_d6da_11e9_9184_309c23131503row3_col4\" class=\"data row3 col4\" >0.76</td>\n",
       "                        <td id=\"T_95779fa8_d6da_11e9_9184_309c23131503row3_col5\" class=\"data row3 col5\" >0.32</td>\n",
       "            </tr>\n",
       "            <tr>\n",
       "                        <th id=\"T_95779fa8_d6da_11e9_9184_309c23131503level0_row4\" class=\"row_heading level0 row4\" >grade</th>\n",
       "                        <td id=\"T_95779fa8_d6da_11e9_9184_309c23131503row4_col0\" class=\"data row4 col0\" >0.67</td>\n",
       "                        <td id=\"T_95779fa8_d6da_11e9_9184_309c23131503row4_col1\" class=\"data row4 col1\" >0.37</td>\n",
       "                        <td id=\"T_95779fa8_d6da_11e9_9184_309c23131503row4_col2\" class=\"data row4 col2\" >0.61</td>\n",
       "                        <td id=\"T_95779fa8_d6da_11e9_9184_309c23131503row4_col3\" class=\"data row4 col3\" >0.76</td>\n",
       "                        <td id=\"T_95779fa8_d6da_11e9_9184_309c23131503row4_col4\" class=\"data row4 col4\" >1</td>\n",
       "                        <td id=\"T_95779fa8_d6da_11e9_9184_309c23131503row4_col5\" class=\"data row4 col5\" >0.45</td>\n",
       "            </tr>\n",
       "            <tr>\n",
       "                        <th id=\"T_95779fa8_d6da_11e9_9184_309c23131503level0_row5\" class=\"row_heading level0 row5\" >yr_built</th>\n",
       "                        <td id=\"T_95779fa8_d6da_11e9_9184_309c23131503row5_col0\" class=\"data row5 col0\" >0.054</td>\n",
       "                        <td id=\"T_95779fa8_d6da_11e9_9184_309c23131503row5_col1\" class=\"data row5 col1\" >0.16</td>\n",
       "                        <td id=\"T_95779fa8_d6da_11e9_9184_309c23131503row5_col2\" class=\"data row5 col2\" >0.43</td>\n",
       "                        <td id=\"T_95779fa8_d6da_11e9_9184_309c23131503row5_col3\" class=\"data row5 col3\" >0.32</td>\n",
       "                        <td id=\"T_95779fa8_d6da_11e9_9184_309c23131503row5_col4\" class=\"data row5 col4\" >0.45</td>\n",
       "                        <td id=\"T_95779fa8_d6da_11e9_9184_309c23131503row5_col5\" class=\"data row5 col5\" >1</td>\n",
       "            </tr>\n",
       "    </tbody></table>"
      ],
      "text/plain": [
       "<pandas.io.formats.style.Styler at 0x1dcf565d898>"
      ]
     },
     "execution_count": 46,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Mengetahui nilai korelasi dari independent variable dan dependent variable\n",
    "df.corr().style.background_gradient().set_precision(2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- Dari tabel korelasi diatas, dapat dilihat bahwa sqft_living mempunyai hubungan linear positif yang sangat kuat dengan price jika dibandingkan yang lain.\n",
    "- Nilai korelasi yr_built hampir mendekati nol yang menandakan bahwa usia rumah tidak mempengaruhi pada harga rumah."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Modelling"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>price</th>\n",
       "      <th>bedrooms</th>\n",
       "      <th>bathrooms</th>\n",
       "      <th>sqft_living</th>\n",
       "      <th>grade</th>\n",
       "      <th>yr_built</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>221900.0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>1180</td>\n",
       "      <td>7</td>\n",
       "      <td>1955</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>538000.0</td>\n",
       "      <td>3</td>\n",
       "      <td>2</td>\n",
       "      <td>2570</td>\n",
       "      <td>7</td>\n",
       "      <td>1951</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>180000.0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>770</td>\n",
       "      <td>6</td>\n",
       "      <td>1933</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>604000.0</td>\n",
       "      <td>4</td>\n",
       "      <td>3</td>\n",
       "      <td>1960</td>\n",
       "      <td>7</td>\n",
       "      <td>1965</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>510000.0</td>\n",
       "      <td>3</td>\n",
       "      <td>2</td>\n",
       "      <td>1680</td>\n",
       "      <td>8</td>\n",
       "      <td>1987</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      price  bedrooms  bathrooms  sqft_living  grade  yr_built\n",
       "0  221900.0         3          1         1180      7      1955\n",
       "1  538000.0         3          2         2570      7      1951\n",
       "2  180000.0         2          1          770      6      1933\n",
       "3  604000.0         4          3         1960      7      1965\n",
       "4  510000.0         3          2         1680      8      1987"
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Recall data kita\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Pertama, buat variabel x dan y\n",
    "x = df.drop(columns='price')\n",
    "y = df['price']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Kedua, kita split data kita menjadi training and testing dengan porsi 80:20\n",
    "x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(17290, 5)\n",
      "(17290,)\n",
      "(4323, 5)\n",
      "(4323,)\n"
     ]
    }
   ],
   "source": [
    "#Cek shape dari data training dan testing\n",
    "print(x_train.shape)\n",
    "print(y_train.shape)\n",
    "print(x_test.shape)\n",
    "print(y_test.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Ketiga, kita bikin object linear regresi\n",
    "lin_reg = LinearRegression()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)"
      ]
     },
     "execution_count": 52,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Keempat, train the model menggunakan training data yang sudah displit\n",
    "lin_reg.fit(x_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[-53061.75464279  64658.55790616    188.90926343 131290.89536823\n",
      "  -3969.55831454]\n",
      "7031568.245717673\n"
     ]
    }
   ],
   "source": [
    "#Kelima, cari tau nilai slope/koefisien (m) dan intercept (b)\n",
    "print(lin_reg.coef_)\n",
    "print(lin_reg.intercept_)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>features</th>\n",
       "      <th>coef_value</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>bedrooms</td>\n",
       "      <td>-53061.754643</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>bathrooms</td>\n",
       "      <td>64658.557906</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>sqft_living</td>\n",
       "      <td>188.909263</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>grade</td>\n",
       "      <td>131290.895368</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>yr_built</td>\n",
       "      <td>-3969.558315</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      features     coef_value\n",
       "0     bedrooms  -53061.754643\n",
       "1    bathrooms   64658.557906\n",
       "2  sqft_living     188.909263\n",
       "3        grade  131290.895368\n",
       "4     yr_built   -3969.558315"
      ]
     },
     "execution_count": 54,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Kita coba buat kedalam dataframe agar kebih rapi\n",
    "coef_dict = {\n",
    "    'features': x.columns,\n",
    "    'coef_value':lin_reg.coef_\n",
    "}\n",
    "coef = pd.DataFrame(coef_dict, columns=['features', 'coef_value'])\n",
    "coef"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- dari nilai m dan b diatas, kalau dimasukan ke dalam rumus menjadi:\n",
    "Y = -53061.75x1 + 64658.56x2 + 188.91x3 + 131290.89x4 - 3969.56x5 + 7031568"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred = lin_reg.predict(x_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.6125113286941102"
      ]
     },
     "execution_count": 56,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Keenam, kita cari tahu accuracy score dari model kita menggunakan testing data yang sudah displit\n",
    "lin_reg.score(x_test, y_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- Model kita mendapatkan accuracy score sebesar 61.13%"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Prediction\n",
    "- Yuk kita prediksi harga rumah sesuai dengan permintaan Joko dengan kriteria sebagai berikut:\n",
    "    1. bedrooms = 3\n",
    "    2. bathrooms = 2\n",
    "    3. sqft_living = 1800 sqft\n",
    "    4. grade = 7\n",
    "    5. yr_built = 1990"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([361351.99342265])"
      ]
     },
     "execution_count": 57,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Prediksi harga rumah idaman Joko\n",
    "lin_reg.predict([[3,2,1800,7,1990]])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- Yeay! Harga rumah idaman Joko dan istirnya adalah sekitar 361351 US$"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
