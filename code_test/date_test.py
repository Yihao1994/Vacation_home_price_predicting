from datetime import datetime
import time

def date_transfer(date, Base):
    
    sec_for_a_day = 24*60*60
    
    # The date transferred to sec
    date = datetime.strptime(date, '%Y-%m-%d')
    un_time_date = time.mktime(date.timetuple())
    
    # The base transferred to sec
    Base = datetime.strptime(Base, '%Y-%m-%d')
    un_time_Base = time.mktime(Base.timetuple())
    
    day_of_date = (un_time_date - un_time_Base)/sec_for_a_day
    
    return day_of_date

Base = '2016-01-01'
date = '2016-02-10'

day_of_date = date_transfer(date, Base)
print(day_of_date)





        
        BST = xgb.XGBRegressor(seed = 1850)
        BST.fit(X_train, y_train.ravel(), verbose = True)
        
        # R^2 (coefficient of determination) regression score function
        # Training score
        preds_train = BST.predict(X_train)
        
        
        # Testing score
        preds_test = BST.predict(X_test)
        print('Testing r2 score for Group %d: ' % (nr_cluster - 1))
        print(metrics.r2_score(y_test, preds_test))
        print('')
        