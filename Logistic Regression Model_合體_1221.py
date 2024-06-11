import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from matplotlib.font_manager import fontManager
for i in sorted(fontManager.get_font_names()):      # 使用 fontManager 物件檢視系統資料夾中的字型
    print(i)
matplotlib.rc('font', family='Microsoft JhengHei')  # 將繪圖字型改為微軟正黑體


# 讀取 csv 檔，我把財務變數跟類別變數分開讀取
df1 = pd.read_csv("D:\Quantitative\Quantitative_1222\羅吉斯迴歸模型樣本_20231208_財務變數.csv", index_col=0)
df2 = pd.read_csv("D:\Quantitative\Quantitative_1222\羅吉斯迴歸模型樣本_20231208_類別變數.csv", index_col=0)

# 將財務變數每一欄位遺漏值以平均數填滿
df1.fillna(df1.mean(), inplace=True)

# 檢查是否還有遺漏值
df1.isnull().sum()
df2.isnull().sum()

# 把 Y 刪除，另存成財務(finance)、類別(category)，df_x_fin、df_x_cat 兩變數
df_x_fin = df1.drop(columns = ['Y'])
df_x_cat = df2.drop(columns = ['Y'])

# 將財務變數標準化
columns_list = df_x_fin.columns.tolist()
for column in columns_list:
    mean_value = df_x_fin[column].mean()
    std_value = df_x_fin[column].std()
    df_x_fin[column] = (df_x_fin[column] - mean_value) / std_value
df_x_fin = np.clip(df_x_fin, -2, 2)

# 結合財務變數與類別變數，設定為自變數 X
X = pd.concat([df_x_fin, df_x_cat], axis = 1)
y = df1['Y']

# 為自變數 X 添加截距項
X = sm.add_constant(X)

# 將數據拆分為訓練集8：驗測集2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=16, stratify=y)
'''
X_train.to_csv("D:\Quantitative\X_train.csv")
X_test.to_csv("D:\Quantitative\X_test.csv")
y_train.to_csv("D:\Quantitative\y_train.csv")
y_test.to_csv("D:\Quantitative\y_test.csv")
'''


''' 單因子分析，將所有自變數逐一對 Y 進行羅吉斯迴歸，並取得 P-value '''
pvalue_results = pd.DataFrame(columns = ['Variable', 'P-value'])
# 逐一將 X 對 Y 進行羅吉斯迴歸
for column in X.columns[1:]:  # 第一欄為截距項，從第二欄開始
    model = sm.Logit(y, X[[column]])
    result = model.fit()
    p_value = result.pvalues[column]
    pvalue_results = pvalue_results.append({'Variable': column, 'P-value': p_value}, ignore_index=True)

# 印出 P-value < 0.1 之自變數
significant_results = pvalue_results[pvalue_results['P-value'] < 0.1]
print(significant_results)
df_x_significant = X_train.loc[:, significant_results['Variable']]
#pvalue_results.to_csv("D:\Quantitative\Quantitative_1222\P-value.csv", index=False)

''' 建立相關係數矩陣 '''
# 建立相關係數矩陣以分析變數間相關性，將相關係數 > 0.5 或 < -0.5 的變數挑一個留下
corr = df_x_significant.corr()
#corr.to_csv("D:\Quantitative\Quantitative_1222\Corr.csv")

# 把留下的變數找回來
columns_to_keep = ['營業利益率', '稅後淨利率', '現金流量比率', '有息負債利率', 
                   '稅率', '營業毛利成長率', '營業利益成長率', '稅後淨利成長率', 
                   '總負債 / 總淨值', '或有負債 / 淨值', '稅前純益 / 實收資本',
                   '固定資產週轉次數', '有投保董監責任險', 
                   '3 年內 CPA 有異動', '董事長內部化', '監察人內部化', 
                   '3 年內董事長有異動', '3 年內總經理有異動', 
                   '3 年內財務主管有異動', '3 年內發言人有異動', 
                   '3 年內內部稽核有異動', '四大會計師事務所簽證', '存貨週轉率大於 8.515']
df_x_significant_keep = df_x_significant[columns_to_keep]


''' 逐步迴歸 Stepwise Regression '''
# 開始逐步迴歸
# Step 1. 添加截距項
X_train = sm.add_constant(df_x_significant_keep)

# Step 2. 建立逐步迴歸 Function
def stepwise_selection(X, y, initial_list=[], threshold_in=0.4, threshold_out=0.5, verbose=True):
    included = list(initial_list)
    while True:
        changed = False
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded, dtype=float)
        for new_column in excluded:
            model = sm.Logit(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))
        model = sm.Logit(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()
        if worst_pval > threshold_out:
            changed = True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included

# Step 3. 將訓練集丟進去跑逐步迴歸
stepwise_result = stepwise_selection(X_train, y_train)

# Step 4. 印出該選擇的變數
print("Selected features:", stepwise_result)


# 將 Model 1, 2, 3 所納入之變數設定完畢
# Model 1 之自變數
result1 = stepwise_result
remove_element1 = ['稅率', '董事長內部化', '3 年內董事長有異動', 
                   '3 年內財務主管有異動', '3 年內 CPA 有異動']
for element in remove_element1:
    result1.remove(element)

# Model 2 之自變數
result2 = list(result1)
new_element2 = ['稅後淨利成長率']
result2.extend(new_element2)

# Model 3 之自變數
result3 = list(result2)
new_element3 = ['稅前純益 / 實收資本']
result3.extend(new_element3)

# 建立最終自變數 X 的 DataFrame
X_train_final_1 = X_train[result1]
X_test_final_1 = X_test[result1]
X_train_final_2 = X_train[result2]
X_test_final_2 = X_test[result2]
X_train_fianl_3 = X_train[result3]
X_test_final_3 = X_test[result3]


''' 羅吉斯迴歸模型、ROC Curve、Confusion Matrix 的 Function '''
def Build_Logistic_Model(i, X_train_final, X_test_final, y_train):
    
    ''' 羅吉斯迴歸 Logistic Regression '''
    # 添加截距項
    X_train_final = sm.add_constant(X_train_final)

    # 建立 Logistic Model
    model = sm.Logit(y_train, X_train_final)

    # 訓練擬合模型並印出結果
    result = model.fit()
    print(result.summary())


    ''' 訓練集 ROC Curve '''
    # 將訓練集丟入取得 Logistic Model 之預測機率
    y_prob = result.predict(X_train_final)

    # 計算訓練集 ROC Curve
    fpr, tpr, thresholds = roc_curve(y_train, y_prob)

    # 計算訓練集 AUC, Area Under the Curve
    auc = roc_auc_score(y_train, y_prob)

    # 繪製訓練集 ROC 曲線
    plt.figure(dpi=600)
    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{i} 號模型訓練集 ROC Curve', fontweight='bold')
    plt.legend()
    plt.show()


    ''' 測試集 ROC Curve '''
    # 將測試集丟入取得 Logistic Model 之預測機率
    y_test_prob = result.predict(X_test_final)

    # 計算測試集 ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)

    # 計算測試集 AUC, Area Under the Curve
    auc = roc_auc_score(y_test, y_test_prob)

    # 繪製 ROC 曲線
    plt.figure(dpi=600)
    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{i} 號模型測試集 ROC Curve', fontweight='bold')
    plt.legend()
    plt.show()


    ''' 建立訓練 Confusion Matrix 混淆矩陣
        Model 1: 0.13, Model 2: 0.2, Model 3: 0.17 '''
    # 門檻值 0.0-1.0
    for i in range(0, 11):
        i /= 10.0
        # 設定門檻值，將預測出來之機率轉為 0 或 1
        y_pred_binary = (y_prob >= i).astype(int)

        # 計算混淆矩陣
        cm_prob = confusion_matrix(y_train, y_pred_binary)
        TN = cm_prob[0][0]
        FP = cm_prob[0][1]
        FN = cm_prob[1][0]
        TP = cm_prob[1][1]

        # 顯示混淆矩陣
        print(f'Cut-off: {i}')
        print('Confusion Matrix:')
        print(cm_prob)
        print(FN / (FN + TP))
        print('FP + FN =', FP + FN)
        print('Accuracy:', accuracy_score(y_train, y_pred_binary))
        print()

    # 門檻值 0.01-0.20
    for i in range(0, 21):
        i /= 100.0
        # 設定門檻值，將預測出來之機率轉為 0 或 1
        y_pred_binary = (y_prob >= i).astype(int)

        # 計算混淆矩陣
        cm_prob = confusion_matrix(y_train, y_pred_binary)
        TN = cm_prob[0][0]
        FP = cm_prob[0][1]
        FN = cm_prob[1][0]
        TP = cm_prob[1][1]

        # 顯示混淆矩陣
        print(f'Cut-off: {i}')
        print('Confusion Matrix:')
        print(cm_prob)
        print(FN / (FN + TP))
        print('FP + FN =', FP + FN)
        print('Accuracy:', accuracy_score(y_train, y_pred_binary))
        print()


''' Confusion Matrix 繪圖的 Function '''
def Draw_Confusion_Matrix(i, cut_off, X_train_final, y_train):
    # 添加截距項
    X_train_final = sm.add_constant(X_train_final)

    # 建立 Logistic Model
    model = sm.Logit(y_train, X_train_final)

    # 訓練擬合模型並印出結果
    result = model.fit()
    
    # 將訓練集丟入取得 Logistic Model 之預測機率
    y_prob = result.predict(X_train_final)

    # 繪製混淆矩陣圖片
    y_pred_binary = (y_prob >= float(cut_off)).astype(int)
    cm = confusion_matrix(y_train, y_pred_binary)

    # 設置標註字體大小
    annot_font_size = 18
    annot_kws = {'size': annot_font_size}

    plt.figure(dpi=600)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=['預測為 0', '預測為 1'], 
                yticklabels=['實際為 0', '實際為 1'], 
                annot_kws=annot_kws)
    plt.title(f'{i} 號模型訓練集之 Confusion Matrix', fontweight='bold')
    plt.xlabel('預測值')
    plt.ylabel('實際值')
    plt.show()

Build_Logistic_Model(1, X_train_final_1, X_test_final_1, y_train)
Draw_Confusion_Matrix(1, 0.14, X_train_final_1, y_train)
Build_Logistic_Model(2, X_train_final_2, X_test_final_2, y_train)
Draw_Confusion_Matrix(2, 0.2, X_train_final_2, y_train)
Build_Logistic_Model(3, X_train_fianl_3, X_test_final_3, y_train)
Draw_Confusion_Matrix(3, 0.2, X_train_fianl_3, y_train)



'''
混淆矩陣的解讀：[[TN, FP]
               [FN, TP]]

TN, True Negative; FP, False Positive
FN, False Negative; TP, True Positive

Accuracy = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
'''