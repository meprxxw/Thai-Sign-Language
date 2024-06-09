# Thai-Sign-Language 💭
This project aims to develop a Thai Sign Language detection using a Transformer-based model.

## code </>
### Model Files
- **model.tflite**: This model was trained using the `TSL-transformer-training.ipynb` notebook.
- **model-withflip.tflite**: This model was trained with additional data augmentation (flipping) using the `TSL-withflip-transformer-training.ipynb` notebook.

### Data Processing
- **parquetdata_visualize.ipynb**: This notebook explains how the data was processed and visualized before being used for model training.

## app 🚀
You can try the thai sign language detection app through this link:

Link🔗 : [Thai Sign Language Detection App](https://thai-sign-language-r39hfekjqvt7ykmfh2nj5y.streamlit.app/)

Alternatively, you can clone the repository and run the app locally to try real-time thai sign language detection app:
```sh
streamlit run real_time_app.py
```

## dataset 🗃️
- numpy file : https://github.com/Annerez/Mediapipe_ThaiHandSign?fbclid=IwAR1OPuXqOO-M6qd8OI24fogd3GSZ_E1hA42iUwLtAGltuCNthxz7OmO1Ltg 
- parquet file : https://drive.google.com/drive/folders/1f2uqMtYWMfLyeFzRrC0lwu0jG7ItVYsu?usp=sharing
- parquet file with flip : https://drive.google.com/drive/folders/1SGlHoKKlsNQA-vk4_sXyDML6Lo-PX5YR?usp=sharing

## more info
For a detailed walkthrough of the project, including how to improve it, you can read my article on Medium: https://medium.com/@feat.meprxxw/thai-sign-language-detection-07a3d864a3b4
