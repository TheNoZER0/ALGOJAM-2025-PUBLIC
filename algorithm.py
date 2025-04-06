import numpy as np
from statsmodels.tsa.api import VAR, ARIMA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# Custom trading Algorithm
class Algorithm():

    ########################################################
    # NO EDITS REQUIRED TO THESE FUNCTIONS
    ########################################################
    # FUNCTION TO SETUP ALGORITHM CLASS
    def __init__(self, positions):
       
        self.data = {}
      
        self.positionLimits = {}
      
        self.day = 0
       
        self.positions = positions
        self.var_model = None
        self.scaler = StandardScaler()
        self.lookback = 1.1
        self.threshold = 0.0005 
        self.lag_order = 1
        self.var_instruments = ['Fried Chicken', 'Raw Chicken', 'Secret Spices']
        self.totalDailyBudget = 600000 

    
    def get_current_price(self, instrument):
        return self.data[instrument][-1]
    ########################################################

    # RETURN DESIRED POSITIONS IN DICT FORM
    def get_positions(self):
        # Get current position
        currentPositions = self.positions
        # Get position limits
        positionLimits = self.positionLimits

        # Declare a store for desired positions
        desiredPositions = {}
        # Initialise to hold positions for all instruments
        for instrument, positionLimit in positionLimits.items():
            desiredPositions[instrument] = 0

        # IMPLEMENT CODE HERE TO DECIDE WHAT POSITIONS YOU WANT 
        #######################################################################
        desiredPositions["UQ Dollar"] = self.get_uq_dollar_position(currentPositions["UQ Dollar"], positionLimits["UQ Dollar"])
        desiredPositions['Dawg Food'] = self.get_dwgFood_position()
        desiredPositions["Quantum Universal Algorithmic Currency Koin"] = self.get_quack_position(currentPositions["Quantum Universal Algorithmic Currency Koin"], positionLimits["Quantum Universal Algorithmic Currency Koin"])
        desiredPositions["Goober Eats"] = self.get_goober_eats_position()
        
        # apply Purple Elixir strategy
        self.get_prplelixr_position(desiredPositions, positionLimits)
        #desiredPositions["Fintech Token"] = self.get_fintech_position(currentPositions["Fintech Token"], positionLimits["Fintech Token"])
        self.apply_arima_model("Fintech Token", positionLimits, desiredPositions, p=2, d=1, q=1)
        # Apply Regression Model for Fried Chicken
        self.apply_regression_model(positionLimits, desiredPositions)
        # Apply ARIMA for Secret Spices and raw_chicken
        self.apply_arima_model("Secret Spices", positionLimits, desiredPositions)
        self.apply_arima_model("Raw Chicken", positionLimits, desiredPositions)
        self.apply_arima_model("Rare Watch", positionLimits, desiredPositions, p=2, d=1, q=1)
        desiredPositions = self.scale_positions(desiredPositions, currentPositions)
        return desiredPositions

    ########################################################
    # HELPER METHODS
        
    def apply_regression_model(self, positionLimits, desiredPositions):
        FC = self.get_current_price("Fried Chicken")
        RC = self.get_current_price("Raw Chicken")
        SS = self.get_current_price("Secret Spices")
        if(-129.04616223805*FC +21.6126341844*RC  + SS > -194.2325350693):
            desiredPositions["Fried Chicken"] = positionLimits["Fried Chicken"]
            desiredPositions["Secret Spices"] = -positionLimits["Secret Spices"]
            desiredPositions["Raw Chicken"] = -positionLimits["Raw Chicken"]
            Direction = -1

        if(-129.04616223805*FC +21.6126341844*RC  + SS < -194.2325350693):
            desiredPositions["Fried Chicken"] = -positionLimits["Fried Chicken"]
            desiredPositions["Secret Spices"] = positionLimits["Secret Spices"]
            desiredPositions["Raw Chicken"] = positionLimits["Raw Chicken"]
            Direction = 1

    # def apply_arima_model(self, instrument, positionLimits, desiredPositions, p=0, d=0, q=1):

    #     if self.day < 10: 
    #         p, d, q = 2, 0, 1

    #     if self.day >= self.lookback:
    #         data = np.array(self.data[instrument])
    #         model = ARIMA(data, order=(p, d, q))
    #         model_fit = model.fit()
    #         forecast = model_fit.forecast(steps=1)[0]

    #         current_price = self.get_current_price(instrument)
    #         price_diff = forecast - current_price
    #         if abs(price_diff) > self.threshold * current_price:
    #             # Use full position limit
    #             position = positionLimits[instrument] if price_diff > 0 else -positionLimits[instrument]
    #             desiredPositions[instrument] = position

   
    def adjust_positions_for_budget(self, desiredPositions, positionLimits):
        total_value = 0
        maxInventoryValue = 599997
        prices = {inst: self.get_current_price(inst) for inst in desiredPositions}

        for inst, pos in desiredPositions.items():
            if inst in prices:
                total_value += abs(pos * prices[inst])

                if total_value > self.totalDailyBudget:
                    scaling_factor = self.totalDailyBudget / total_value
                    for inst_inner in desiredPositions:
                        if inst_inner in prices:
                            desiredPositions[inst_inner] = int(desiredPositions[inst_inner] * scaling_factor)

                if total_value < self.totalDailyBudget:
                    if "Secret Spices" in prices and prices["Secret Spices"] > 1e-9:
                        numberofSecretSpices = int((self.totalDailyBudget - total_value) / prices["Secret Spices"])
                        current_direction = getattr(self, 'Direction', 1)
                        if current_direction == 0: current_direction = 1

                        if(abs(int(current_direction*numberofSecretSpices)) > 200):
                            if "Secret Spices" in positionLimits:
                                limit = positionLimits["Secret Spices"]
                                final_pos_target = int(current_direction*limit)
                                desiredPositions["Secret Spices"] = max(-limit, min(limit, final_pos_target))
                                desiredPositions["Secret Spices"] = int(desiredPositions["Secret Spices"])
                        else:
                            final_pos_target = int(current_direction*numberofSecretSpices)
                            if "Secret Spices" in positionLimits:
                                limit = positionLimits["Secret Spices"]
                                desiredPositions["Secret Spices"] = max(-limit, min(limit, final_pos_target))
                                desiredPositions["Secret Spices"] = int(desiredPositions["Secret Spices"])

        if total_value > maxInventoryValue:
            excessValue = total_value - maxInventoryValue
            for instrument in ['Fintech Token', 'Quantum Universal Algorithmic Currency Koin', 'UQ Dollar', 'Raw Chicken', 'Secret Spices', 'Fried Chicken', 'Goober Eats', 'Purple Elixir', 'Dawg Food']:
                if instrument in desiredPositions and instrument in prices and desiredPositions[instrument] != 0 and excessValue > 1e-6:
                    instrumentValue = abs(desiredPositions[instrument] * prices[instrument])
                    current_pos = desiredPositions[instrument]

                    if instrumentValue <= excessValue + 1e-6:
                        excessValue -= instrumentValue
                        desiredPositions[instrument] = 0
                    else:
                        reductionAmountUnits = excessValue / prices[instrument]
                        reduction_int_units = int(reductionAmountUnits)
                        if reduction_int_units > 0:
                            desiredPositions[instrument] -= np.sign(current_pos) * reduction_int_units
                        excessValue = 0

                if excessValue < 1e-6:
                    break

        for inst in desiredPositions:
            desiredPositions[inst] = int(desiredPositions.get(inst, 0))

        return desiredPositions
    

    def scale_positions(self, desiredPositions, currentPositions):
        total_pos_value, prices_current, pos_values = self.calc_current_total_trade_val(desiredPositions, currentPositions)
        # if the total trade value is greater than the total daily budget, scale down the trade values for tokens
        if total_pos_value > self.totalDailyBudget:
            # find value we need to reduce by
            reduction_val = total_pos_value - self.totalDailyBudget
            # first reduce tokens because they are inneficient, but big size
            # find amount to reduce
            reduction_Tokens = int(reduction_val/prices_current["Rare Watch"])
            # if trades are positive, reduce trades, otherwise increase trades
            if pos_values["Rare Watch"] > 0:
                desiredPositions["Rare Watch"] -= min(reduction_Tokens, desiredPositions["Rare Watch"])
            else:
                desiredPositions["Rare Watch"] += min(reduction_Tokens, desiredPositions["Rare Watch"])               

            total_pos_value, prices_current, pos_values = self.calc_current_total_trade_val(desiredPositions, currentPositions)
            # if we are under the budget, return desired positions
            if total_pos_value <= self.totalDailyBudget:
                return desiredPositions
         
            for inst in ['Fintech Token', 'Quantum Universal Algorithmic Currency Koin', 'UQ Dollar', 'Raw Chicken', 'Secret Spices', 'Fried Chicken' 'Goober Eats', 'Purple Elixir', 'Dawg Food']:
                # calculate required to reduce
                reduction_val = total_pos_value - self.totalDailyBudget
                # find amount to reduce
                reduction_inst = int(reduction_val/prices_current[inst])+1
                # reduce the desired positions
                if pos_values[inst] > 0:
                    desiredPositions[inst] -= min(reduction_inst, desiredPositions[inst])
                else:
                    desiredPositions[inst] += min(reduction_inst, -desiredPositions[inst])

                total_pos_value, prices_current, pos_values = self.calc_current_total_trade_val(desiredPositions, currentPositions)
                
                # if we are under the budget, break
                if total_pos_value <= self.totalDailyBudget:
                    return desiredPositions
        return desiredPositions
                

    def calc_current_total_trade_val(self, desiredPositions, currentPositions):
        # get prices for all instruments as a dictionary
        prices_current = {inst: self.get_current_price(inst) for inst in desiredPositions}
        # dict of trade values which is the trade amount multiplied by the current price
        pos_values = {inst: abs(desiredPositions[inst] * prices_current[inst]) for inst in desiredPositions}
        # total trade value is the sum of all trade values
        total_pos_value = sum(pos_values.values())

        return total_pos_value, prices_current, pos_values

    def get_prplelixr_position(self, desiredPositions, positionLimits):
        elixr_df = pd.DataFrame(self.data["Purple Elixir"])
        quack_df = pd.DataFrame(self.data["Quantum Universal Algorithmic Currency Koin"])

        elixr_df['EMA'] = elixr_df[0].ewm(span=5, adjust=False).mean()
        quack_df['EMA'] = quack_df[0].ewm(span=5, adjust=False).mean()

        elixr_df['EMA25'] = elixr_df[0].ewm(span=25, adjust=False).mean()
        elixr_df['Cross'] = elixr_df['EMA'] - elixr_df['EMA25']

        price_drink = self.data['Purple Elixir'][-1]
        price_quack = self.data['Quantum Universal Algorithmic Currency Koin'][-1]

        ema_drink = elixr_df['EMA'].iloc[-1]
        ema_quack = quack_df['EMA'].iloc[-1]
        
        cross_signal = elixr_df['Cross'].iloc[-1]

        theo = ema_drink -0.025*ema_quack + 0.055*np.sign(cross_signal)*(abs(cross_signal)**(1/4))

        if price_drink > theo:
            desiredPositions["Purple Elixir"] = -positionLimits["Purple Elixir"]
        else:
            desiredPositions["Purple Elixir"] = positionLimits["Purple Elixir"]

    def get_goober_eats_position(self):
        goober_df = pd.DataFrame(self.data["Goober Eats"])
        goober_df['EMA'] = goober_df[0].ewm(span=13, adjust=False).mean()
        # Buy if the price is above the 5 day EMA
        price = self.data['Goober Eats'][-1]
        ema = goober_df['EMA'].iloc[-1]
        limit = self.positionLimits["Goober Eats"]
        if price > ema:
            desiredPositions = -limit
        else:
            desiredPositions = limit
        return desiredPositions

    def get_uq_dollar_position(self, currentPosition, limit):
        avg = sum(self.data["UQ Dollar"][-4:]) / 4
        price = self.get_current_price("UQ Dollar")
        diff = avg - price
        boundary = max(self.data["UQ Dollar"]) - avg

        if diff > 0.24:
            delta = limit * 2
        elif diff < -0.24:
            delta = -2 * limit
        else:
            delta = 0

        if currentPosition + delta > limit:
            desiredPosition = limit
        elif currentPosition + delta < -1 * limit:
            desiredPosition = -1 * limit
        else:
            desiredPosition = currentPosition + delta

        return desiredPosition

    def get_quack_position(self, currentPosition, limit):
        avg = sum(self.data["Quantum Universal Algorithmic Currency Koin"][-10:]) / 10
        price = self.get_current_price("Quantum Universal Algorithmic Currency Koin")

        if price < 2.2:
            desiredPosition = limit
        elif price > 2.45:
            desiredPosition = -1 * limit
        else:
            desiredPosition = currentPosition

        return desiredPosition
    
    def get_dwgFood_position(self):
        dwgFood_df = pd.DataFrame(self.data["Dawg Food"])
        dwgFood_df['EMA5'] = dwgFood_df[0].ewm(span=2, adjust=False).mean()
        # Buy if the price is above the 5 day EMA
        price = self.data['Dawg Food'][-1]
        ema = dwgFood_df['EMA5'].iloc[-1]
        if price > ema:
            desiredPositions = -self.positionLimits["Dawg Food"]
        else:
            desiredPositions = self.positionLimits["Dawg Food"]
        return desiredPositions


    def apply_arima_model(self, instrument, positionLimits, desiredPositions, p=1, d=1, q=1):
        if self.day >= self.lookback:
            data = np.array(self.data[instrument])
            model = ARIMA(data, order=(p, d, q))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=1)[0]

            current_price = self.get_current_price(instrument)
            price_diff = forecast - current_price
            if abs(price_diff) > self.threshold * current_price:
                # Use full position limit
                position = positionLimits[instrument] if price_diff > 0 else -positionLimits[instrument]
                desiredPositions[instrument] = position
    
    def get_token_position(self, currentPosition, limit):

        step = 35

        if self.day < 10:
            return currentPosition

        # Split the list into two halves
        first_half = self.data["Fintech Token"][-10:-5]
        second_half = self.data["Fintech Token"][-5:]

        # Calculate the gradients for each half
        first_grad = self.calculate_gradient(first_half)
        second_grad = self.calculate_gradient(second_half)

        lim = 18

        # Going from a stable section to jumping up
        if abs(first_grad) < lim and second_grad > lim:
            delta = step
        # Going from a stable section to jumping down
        elif abs(first_grad) < lim and second_grad < -1 * lim:
            delta = -1 * step
        else:
            delta = 0

        if currentPosition + delta > limit:
            desiredPosition = limit
        elif currentPosition + delta < -1 * limit:
            desiredPosition = -1 * limit
        else:
            desiredPosition = currentPosition + delta

        return desiredPosition
    

    # Function to calculate linear extrapolation
    def linear_extrapolation(self, values):
        if len(values) < 5:
            return np.nan 
        x = np.arange(5)
        y = values[-6:-1]
        coeffs = np.polyfit(x, y, 1)  # Linear fit (degree 1)
        extrapolated_value = np.polyval(coeffs, 5)  # Extrapolate to the next point
        return extrapolated_value
    
    def calculate_gradient(self, values):
        x = np.arange(5) 
        y = np.array(values)
        # Fit a linear model: y = mx + c
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]

        return m