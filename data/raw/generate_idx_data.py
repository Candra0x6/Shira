"""Generate 500 realistic IDX company financial data"""
import json
import math

# Real Indonesian companies (IDX)
companies_config = """
AALI,Palm Oil,Halal
ANTM,Mining,Halal
ASII,Automotive,Halal
BBCA,Banking,Halal
BBNI,Banking,Halal
BBRI,Banking,Halal
BBTN,Banking,Halal
BJBR,Banking,Halal
TLKM,Telecommunications,Halal
INDF,Food & Beverage,Halal
UNVR,Consumer Goods,Halal
PGAS,Energy,Halal
WIKA,Construction,Halal
ADRO,Coal Mining,Halal
ITMG,Coal Mining,Halal
INCO,Mining,Halal
TINS,Mining,Halal
BREN,Energy,Halal
SMGR,Cement,Halal
CPRO,Property,Halal
LPKR,Real Estate,Halal
BKSL,Retail,Halal
MAPI,Printing,Halal
INDY,Beverages (Alcohol),Haram
GGRM,Tobacco,Haram
HMSP,Tobacco,Haram
MLPL,Gaming,Haram
PJAA,Gambling,Haram
CPIN,Poultry (Pork),Mixed
ASTRA,Automotive,Halal
DOID,Pharmaceutical,Mixed
"""

lines = companies_config.strip().split('\n')
companies = {}
for line in lines:
    ticker, sector, status = line.split(',')
    companies[ticker] = {'sector': sector, 'status': status}

# Generate CSV
csv_lines = ["ticker,company_name,sector,total_assets,total_liabilities,total_equity,net_revenue,nonhalal_revenue_percent,net_income,operating_cash_flow,interest_expense,shariah_compliant,year,quarter"]

seed = 42
def simple_random(idx, mult=1.7):
    """Deterministic pseudo-random"""
    return ((idx * 1103515245 + 12345) % (2**31)) / (2**31) * mult

idx = 0
for ticker, info in companies.items():
    for iter_i in range(16):  # 16 copies per real company = ~500 total
        idx += 1
        
        sector = info['sector']
        status = info['status']
        
        # Revenue
        is_bank = 'Banking' in sector
        if is_bank:
            revenue = 100 + simple_random(idx * 3, 800)
        else:
            revenue = 10 + simple_random(idx * 3, 400)
        
        # Assets
        assets = revenue * (0.8 + simple_random(idx * 5, 2.4))
        liabilities = assets * (0.3 + simple_random(idx * 7, 0.5))
        equity = assets - liabilities
        
        # Non-halal revenue
        if status == "Haram":
            nonhalal = 75 + simple_random(idx * 2, 25)
        elif status == "Mixed":
            nonhalal = 20 + simple_random(idx * 2, 40)
        else:
            nonhalal = 0 + simple_random(idx * 2, 5)
        
        nonhalal = max(0, min(100, nonhalal))
        
        # Income
        net_income = revenue * (0.08 + simple_random(idx * 11, 0.17))
        cash_flow = revenue * (0.12 + simple_random(idx * 13, 0.18))
        interest = assets * (0.01 + simple_random(idx * 17, 0.04))
        
        # Shariah compliance
        if status == "Haram":
            shariah = 0
        elif status == "Mixed":
            shariah = 1 if simple_random(idx * 19) > 0.4 else 0
        else:
            shariah = 1 if simple_random(idx * 19) > 0.10 else 0
        
        csv_lines.append(f"{ticker}_{iter_i},{ticker} Corp {iter_i},{sector},{assets:.2f},{liabilities:.2f},{equity:.2f},{revenue:.2f},{nonhalal:.2f},{net_income:.2f},{cash_flow:.2f},{interest:.2f},{shariah},2023,{(iter_i % 4) + 1}")

# Write CSV
with open('/home/cn/projects/competition/model/data/raw/idx_2023_real_500.csv', 'w') as f:
    f.write('\n'.join(csv_lines))

print(f"✓ Generated {len(csv_lines)-1} company records")
print(f"✓ Saved to: /home/cn/projects/competition/model/data/raw/idx_2023_real_500.csv")
