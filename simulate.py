"""
AI Datacenter 24/7 Base-Load Power Simulation: Nuclear SMR vs. Solar+Storage (GES+BESS)
========================================================================================
Author: Talha Çırak (@talhacirak_energy)
Repository: https://github.com/talhacrak01/ai-datacenter-nuclear-sim
Reference Standards: EPRI AI Power Forecast (2024), U.S. DOE Nuclear Energy Office, BNEF 2025/2026.
"""

import os
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

@dataclass
class HyperscaleAIDatacenter:
    """Techno-economic parameters for a 100,000 GPU AI Training & Inference Cluster."""
    num_gpus: int = 100_000
    gpu_server_tray_tdp_w: float = 1020.0  # NVIDIA H100 SXM5 / B200 weighted board + NVLink TDP
    aux_networking_pct: float = 0.25        # High-density InfiniBand/RoCE, leaf-spine switches, optics
    pue: float = 1.18                       # Power Usage Effectiveness (Direct-to-chip liquid cooling)
    target_availability: float = 0.99999    # Five-nines tier IV data center uptime

    @property
    def it_load_mw(self) -> float:
        """Net computational IT load in Megawatts."""
        raw_it = self.num_gpus * self.gpu_server_tray_tdp_w * (1.0 + self.aux_networking_pct)
        return raw_it / 1e6  # ~127.5 MW

    @property
    def total_grid_demand_mw(self) -> float:
        """Total continuous power drawn from the electrical grid including PUE."""
        return self.it_load_mw * self.pue  # ~150.45 MW (150.5 MW flat base load)

@dataclass
class SMRPlantConfig:
    """Small Modular Reactor (SMR) Configuration (Gen-IV / iPWR standard)."""
    module_rating_mw: float = 81.3        # 2x 81.3 MW = 162.6 MW (~163 MW nameplate)
    num_modules: int = 2
    capacity_factor: float = 0.925        # 92.5% baseload availability
    footprint_donum: float = 25.0         # 25 Dönüm site footprint
    capex_per_kw_usd: float = 6200.0      # Turnkey SMR CAPEX (EPRI/DOE)

    @property
    def total_capacity_mw(self) -> float:
        return self.module_rating_mw * self.num_modules

    @property
    def annual_generation_mwh(self) -> float:
        return self.total_capacity_mw * 8760.0 * self.capacity_factor

@dataclass
class SolarBESSConfig:
    """100% Renewable Off-Grid Equivalent (Solar PV + Battery Energy Storage System)."""
    target_continuous_mw: float = 150.45
    solar_capacity_factor: float = 0.147   # Realized net capacity factor considering winter/losses
    bess_hours_discharge: float = 13.83    # Nighttime continuous discharge hours
    bess_rte: float = 0.865                # Net AC-to-AC Round Trip Efficiency (86.5%)
    solar_footprint_donum_per_mw: float = 15.0 # ~15 dönüm / MW utility tracker installation

    @property
    def required_solar_capacity_mw(self) -> float:
        """Solar overbuild factor required to supply 150.5 MW average day+night."""
        return self.target_continuous_mw / self.solar_capacity_factor

    @property
    def required_bess_capacity_mwh(self) -> float:
        """BESS storage sizing to deliver continuous power during non-solar hours."""
        usable_mwh = self.target_continuous_mw * self.bess_hours_discharge
        return usable_mwh / self.bess_rte

    @property
    def total_land_footprint_donum(self) -> float:
        return self.required_solar_capacity_mw * self.solar_footprint_donum_per_mw

def run_simulation():
    dc = HyperscaleAIDatacenter()
    smr = SMRPlantConfig()
    pv_bess = SolarBESSConfig(target_continuous_mw=dc.total_grid_demand_mw)

    print("=" * 75)
    print(" 100,000 GPU HYPERSCALE AI DATACENTER POWER BENCHMARK ")
    print("=" * 75)
    print(f"IT Computational Load:          {dc.it_load_mw:.2f} MW")
    print(f"PUE (Direct Liquid Cooling):    {dc.pue:.2f}")
    print(f"Total Continuous Grid Demand:   {dc.total_grid_demand_mw:.2f} MW (Flat 24/7/365)")
    print("-" * 75)
    print("SOLUTION A: NUCLEAR SMR (GEN-IV / iPWR)")
    print(f"Installed SMR Capacity:         {smr.total_capacity_mw:.1f} MW ({smr.num_modules} x {smr.module_rating_mw:.0f} MW Modules)")
    print(f"Capacity Factor:                {smr.capacity_factor * 100:.1f}% (Baseload Reliability)")
    print(f"Annual Electricity Output:      {smr.annual_generation_mwh:,.0f} MWh/year")
    print(f"Total Land Footprint:           {smr.footprint_donum:.1f} Dönüm (2.5 Hectares)")
    print("-" * 75)
    print("SOLUTION B: SOLAR PV + BESS STORAGE (7X OVERBUILD)")
    print(f"Required Solar PV Capacity:     {pv_bess.required_solar_capacity_mw:,.1f} MWp (Overbuild Ratio: ~6.8x)")
    print(f"Required Utility BESS Storage:  {pv_bess.required_bess_capacity_mwh:,.1f} MWh (2.41 GWh)")
    print(f"Total Land Footprint:           {pv_bess.total_land_footprint_donum:,.0f} Dönüm (~15.3 km²)")
    print("=" * 75)

    # 1. Generate 24-Hour Dispatch Profile Plot
    hours = np.linspace(0, 24, 288)  # 5-minute resolution
    solar_profile = np.zeros_like(hours)
    for i, h in enumerate(hours):
        if 6.0 <= h <= 18.5:
            # Diurnal solar bell curve with peak ~850 MW
            solar_profile[i] = (np.sin((h - 6.0) / 12.5 * np.pi) ** 1.8) * 850.0

    demand_flat = np.full_like(hours, dc.total_grid_demand_mw)

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=150)
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')

    # Flat DC Demand
    ax.plot(hours, demand_flat, color='#ff3c28', linewidth=3.0, linestyle='--', label=f'AI Datacenter Yükü ({dc.total_grid_demand_mw:.1f} MW Sabit)', zorder=6)

    # Solar PV Generation
    ax.plot(hours, solar_profile, color='#eab308', linewidth=2.5, label=f'1.023 MW Solar GES Üretimi (Pik: 850 MW)', zorder=4)

    # Dispatch Fills
    # Battery Night Discharge
    night_discharge = np.minimum(demand_flat, np.maximum(0, demand_flat - solar_profile))
    ax.fill_between(hours, 0, night_discharge, color='#38bdf8', alpha=0.45, label='BESS Gece Deşarjı (Bataryadan Besleme)', zorder=3)

    # Midday Solar Direct Supply to DC
    solar_direct = np.minimum(demand_flat, solar_profile)
    ax.fill_between(hours, 0, solar_direct, color='#22c55e', alpha=0.35, label='Gündüz Doğrudan Güneş Beslemesi', zorder=2)

    # Midday Surplus (Charging BESS)
    day_surplus = np.maximum(0, solar_profile - demand_flat)
    ax.fill_between(hours, demand_flat, solar_profile, color='#a855f7', alpha=0.30, label='Batarya Şarjı & Şebeke Fazlası', zorder=2)

    # SMR Constant Generation Baseline
    ax.plot(hours, np.full_like(hours, 162.6), color='#06b6d4', linewidth=2.0, linestyle=':', label='Alternatif: 163 MW SMR Nükleer Baz Yük (7/24 Kesintisiz)', zorder=5)

    ax.set_xlim(0, 24)
    ax.set_ylim(0, 950)
    ax.set_xlabel('Günün Saatleri (24 Saatlik Zaman Akışı)', fontsize=12, fontweight='bold', color='#c9d1d9')
    ax.set_ylabel('Elektrik Gücü (MW)', fontsize=12, fontweight='bold', color='#c9d1d9')
    ax.set_title('100.000 GPU Veri Merkezi: 24 Saatlik Şebeke Dengesi (Nükleer SMR vs. GES+BESS)', fontsize=13, fontweight='bold', color='#ffffff', pad=15)
    ax.grid(color='#30363d', linestyle=':', alpha=0.55)
    ax.legend(loc='upper right', framealpha=0.9, facecolor='#161b22', edgecolor='#30363d', fontsize=9.5)

    plt.tight_layout()
    dispatch_img = "ai_nuclear_grid_dispatch.png"
    plt.savefig(dispatch_img, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"[OK] 24h Dispatch Plot saved: {dispatch_img}")

    # 2. Generate Capacity Overbuilding & Land Footprint Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=150)
    fig.patch.set_facecolor('#0d1117')
    for ax in (ax1, ax2):
        ax.set_facecolor('#161b22')
        ax.grid(color='#30363d', linestyle=':', alpha=0.5)

    # Bar 1: Installed Capacity
    categories = ['Nükleer SMR', 'Güneş GES (PV)', 'BESS Batarya']
    capacities = [smr.total_capacity_mw, pv_bess.required_solar_capacity_mw, pv_bess.required_bess_capacity_mwh]
    colors_cap = ['#06b6d4', '#eab308', '#38bdf8']
    bars1 = ax1.bar(categories, capacities, color=colors_cap, width=0.55, edgecolor='#ffffff', linewidth=0.8)
    ax1.set_ylabel('Kurulu Güç / Kapasite (MW / MWh)', fontsize=11, fontweight='bold', color='#c9d1d9')
    ax1.set_title('Kurulu Kapasite İhtiyacı (7 Kat Fazla Güneş)', fontsize=12, fontweight='bold', color='#ffffff', pad=12)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 40, f"{yval:,.0f}", ha='center', va='bottom', fontsize=10, fontweight='bold', color='#ffffff')
    ax1.set_ylim(0, 2700)

    # Bar 2: Land Footprint Comparison
    land_cats = ['Nükleer SMR\n(2 Modül)', 'GES + BESS\n(1.023 MW)']
    lands = [smr.footprint_donum, pv_bess.total_land_footprint_donum]
    colors_land = ['#06b6d4', '#ef4444']
    bars2 = ax2.bar(land_cats, lands, color=colors_land, width=0.50, edgecolor='#ffffff', linewidth=0.8)
    ax2.set_ylabel('Arazi İhtiyacı (Dönüm)', fontsize=11, fontweight='bold', color='#c9d1d9')
    ax2.set_title('Arazi Kullanımı: 25 Dönüm vs 15.345 Dönüm', fontsize=12, fontweight='bold', color='#ffffff', pad=12)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 200, f"{yval:,.0f} Dönüm", ha='center', va='bottom', fontsize=10, fontweight='bold', color='#ffffff')
    ax2.set_ylim(0, 17500)

    plt.tight_layout()
    comparison_img = "capacity_and_land_comparison.png"
    plt.savefig(comparison_img, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"[OK] Comparison Plot saved: {comparison_img}")

if __name__ == "__main__":
    run_simulation()
