import os
import pypistats
import condastats.cli
import pandas
import csv
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np


from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


def daily(data, filename, title):
  print(f"    + daily : {filename}")
  lastmonth = data['date'].dropna().max()[:-3]
  data['rolling'] = data.downloads.rolling(window=7).mean()
  chart=data.plot(x="date", y=["downloads","rolling"], figsize=(
      10, 2), logy=True, title=f"{title} - total on period: {data['downloads'].sum()}")  
  chart.figure.savefig(filename)
  chart.figure.savefig(f"{lastmonth}-{filename}")


def prepare_pypi_data(data):
  # retrieve the new vals
  d = data.drop(["category", "percent"], axis=1)
  d['date'] = d['date'].str[:-3]
  d = d.groupby(['date']).sum()
  return d


def prepare_conda_data(data):
  d = pandas.DataFrame([c for ((a, b), c) in data.items()], index=[
      b for ((a, b), c) in data.items()], columns=['downloads'])
  d.index.name = "date"
  return d


def monthly(d, csvfilename, filename, title, except_last_point=True, degree=4, lw=4, alpha=0.2):
  print(f"    + monthly : {filename}")
  # retrieve the current vals
  vals = dict()
  try:
    with open(csvfilename, newline='') as csvfile:
      for row in csv.reader(csvfile):
        vals[row[0]] = row[1]
  except IOError:
    print("warning:", csvfilename, "not found")
  # We just take the two lasts ...
  for k, v in d[-2:]['downloads'].items():
    vals[k] = str(v)
  # write the new current vals
  with open(csvfilename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for k in sorted(vals.keys()):
      writer.writerow([k, int(float(vals[k]))])

  labs, y = zip(*([(k, int(float(vals[k]))) for k in sorted(vals.keys())]))

  if except_last_point:
    newy = y[-1]
    y = y[:-1]
  x = range(len(y))

  # Linear regression
  linear_regressor = LinearRegression()  # create object for the class
  xx = np.array(x).reshape(-1, 1)
  linear_regressor.fit(xx, np.array(y).reshape(-1, 1)
                       )  # perform linear regression
  Y_predlin = linear_regressor.predict(xx)  # make predictions

  # Polynomial fitting
  model = make_pipeline(PolynomialFeatures(degree), Ridge())
  model.fit(xx, y)
  Y_predpol = model.predict(xx)

  # create the plot
  fig, ax = plt.subplots(figsize=(10, 6))
  ax.plot(x, Y_predlin, color='magenta', linewidth=lw, alpha=alpha)
  ax.plot(xx, Y_predpol, color='gold', linewidth=lw, alpha=alpha)
  ax.scatter(x, y, zorder=10, marker='s')
  if except_last_point:
    ax.scatter([len(x)], [newy])
    ax.text(len(x), newy, str(newy),
            verticalalignment='bottom', horizontalalignment='center',
            size=8)

  ax.grid(True)
  ax.set_xticklabels([labs[int(k)]
                      for k in ax.get_xticks()[0:-1]], size="small")

  ax.set_title(
      f"{title} - linear regression ${linear_regressor.coef_[0][0]:6.2f}.x+{linear_regressor.intercept_[0]:6.2f}$")
  fig.savefig(filename)


def pypi_minor(p, filename, title):
  # Pie chart, where the slices will be ordered and plotted counter-clockwise:

  q = dict(zip(*(p['category'].tolist()[:-1], p['percent'].tolist()[:-1])))
  versions = sorted(p['category'].tolist()[:-1])
  sizes = [float(q[x][:-1]) for x in versions[:-1]]

  fig1, ax1 = plt.subplots()

  wedges, texts, autotexts = ax1.pie(
      sizes, autopct='', shadow=False, startangle=90, wedgeprops=dict(width=0.5))
  # Equal aspect ratio ensures that pie is drawn as a circle.
  ax1.axis('equal')

  ax1.legend(wedges, versions,
             title="minor",
             loc="center left",
             bbox_to_anchor=(0.85, 0, 0.5, 1))

  ax1.set_title(f"{title} (null : {q['null']})")
  fig1.savefig(filename)


def go(pypi_module,
       conda_module,
       do_pypipie=True,
       do_pypi=True,
       do_pypinigthly=True,
       do_conda=True,
       suffix="png"):
  now = datetime.now()  # current date and time
  dt = now.strftime("%m/%d/%Y, %H:%M")
  res = f"{dt} : Begin("
  l = []
  if do_pypipie:
    l.append("pypi pie:"+pypi_module)
  if do_pypi:
    l.append("pypi:"+pypi_module)
  if do_pypinigthly:
    l.append("pypi:"+pypi_module + "-nightly")
  if do_conda:
    l.append("conda:"+conda_module)
  print(res+', '.join(l)+')')

  ###################################################################################
  # PypiPie
  if do_pypipie:
    print("** pypipie")
    data = pypistats.python_minor(pypi_module, format='pandas')
    pypi_minor(
        data, filename=f"{pypi_module}-pypi-minor-pie.{suffixe}", title=f"[{dt}] Python minors")

  ###################################################################################
  # Pypi
  if do_pypi:
    print("** pypi")
    data = pypistats.overall(pypi_module, total=True, format="pandas")
    ##########################
    # without mirrors
    data_without = data.groupby("category").get_group(
        "without_mirrors").sort_values("date")
    daily(data_without, filename=f"{pypi_module}-pypi-daily-without.{suffixe}",
          title=f"[{dt}] {pypi_module} - Pypi, daily, without mirrors")
    monthly(prepare_pypi_data(data_without),
            csvfilename=f"{pypi_module}-pypi-monthly-without.csv",
            filename=f"{pypi_module}-pypi-monthly-without.{suffixe}",
            title=f"{pypi_module} - Pypi monthly, without mirrors")

    ##########################
    # with mirrors
    data_with = data.groupby("category").get_group(
        "with_mirrors").sort_values("date")
    daily(data_with, filename=f"{pypi_module}-pypi-daily-with.{suffixe}",
          title=f"{pypi_module} - Pypi, daily, with mirrors")
    monthly(prepare_pypi_data(data_with),
            csvfilename=f"{pypi_module}-pypi-monthly-with.csv",
            filename=f"{pypi_module}-pypi-monthly-with.{suffixe}",
            title=f"{pypi_module} - Pypi monthly, with mirrors")

  ###################################################################################
  # Pypi-nightly
  if do_pypinigthly:
    print("** pypi nightly")
    datanightly = pypistats.overall(pypi_module+"-nightly",
                                    total=True, format="pandas")
    datanightly_without = datanightly.groupby("category").get_group(
        "without_mirrors").sort_values("date")
    daily(datanightly_without, filename=f"{pypi_module}-nightly-pypi-daily-without.{suffixe}",
          title=f"{pypi_module}-nightly - Pypi, daily, without mirrors")

  ###################################################################################
  # Conda
  if do_conda:
    print("** conda")
    dataconda = condastats.cli.overall([conda_module], monthly=True)
    monthly(prepare_conda_data(dataconda),
            csvfilename=f"{conda_module}-conda-monthly.csv",
            filename=f"{conda_module}-conda-monthly.{suffixe}",
            title=f"{conda_module} - conda monthly",
            except_last_point=False)

  print("End.")


suffixe = "svg"
go(pypi_module="openturns",
   conda_module="openturns",
   do_pypipie=True,
   do_pypi=True,
   do_pypinigthly=False,
   do_conda=True,
   suffix=suffixe)
