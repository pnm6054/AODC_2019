# AODC_2019
Enterprise Category - WeatherRisk &amp; GIGABYTE

- Wind power generation forecast model with django web server (prototype) <br>
using kma's aws, asos data

- 1 hour prediction model <br>
![1 hour prediction](./blog/static/media/mm_data_robust_gen_1.png)

- result
![1 hour prediction result](./blog/static/media/prediction.png)
> input past 72 hours weather data [temp, wind dir, wind speed ...] <br>
> output next 1 hour wind generation output

![72 hours prediction model](./blog/static/media/model.png)
> input past 120 hours data <br>
> output next 72 hour wind generation output

for more **preprocessing** information [description](https://github.com/pnm6054/AODC_2019/blob/master/blog/static/description.html)