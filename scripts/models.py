from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

def get_model(name):
    if name == 'rf':
        return RandomForestRegressor(random_state=42)
    elif name == 'gbr':
        return GradientBoostingRegressor(random_state=42)
    elif name == 'xgb':
        return XGBRegressor(random_state=42, objective='reg:squarederror')
    else:
        raise ValueError("Model name not recognized")

def get_param_grid(name):
    if name == 'rf':
        return {
            'max_depth': [10, 20],
            'min_samples_leaf': [1, 2]
        }
    elif name == 'gbr':
        return {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5],
            'min_samples_split': [2, 5]
        }
    elif name == 'xgb':
        return {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5],
            'min_child_weight': [1, 3]
        }

def train_all_models(X_train, y_train, X_test, y_test):
    model_names = ['rf', 'gbr', 'xgb']
    results = {}

    for name in model_names:
        print(f"\n🔍 Training {name.upper()} with GridSearchCV...")
        model = get_model(name)
        param_grid = get_param_grid(name)

        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        results[name] = {
            'best_params': grid_search.best_params_,
            'mse': mse,
            'model': best_model
        }

        print(f"✅ {name.upper()} done. Best MSE: {mse:.4f}")
        print(f"Best Params: {grid_search.best_params_}")

    return results