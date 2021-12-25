# data_drift_detctor

Implementation of a Machine Learning Operations pipeline, which consists of: model training, data drift detection & data generation to generate data drift. 
Databases being used are:
- https://archive.ics.uci.edu/ml/datasets/bank+marketing
- https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)



אנחנו צריכים להתכנס בסופש על הדברים הבאים:
1. סינתוז של הדאטה - בצורה הבסיסית, גאן כרגע בעדיפות שנייה לא?
2. להחליט על משקול ומודל של הre training
3. לעשות evaluation של הretraining אל מול המודל המקורי - להחליט האם ab testing או מה
4. בדיקות כוללות של הסנתוז והדיטקשן ביחד