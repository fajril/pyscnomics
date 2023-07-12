"""
Developed by:

        - Muhammad Adhim Mulia, M.T. [1]
        - Aditya Dewanto Hartono, Dr. Eng. [1]
        - Fajril Ambia, Ph.D. [2]
        - Iskandar Fahmi, Ph.D. [3]

        Affiliations
        ____________

        [1] Lembaga Pengembangan Inovasi dan Kewirausahaan (LPIK), Institut Teknologi Bandung
            Gedung Energi R7f, Jalan Ganesha 10 Bandung 40132, Indonesia

        [2] Satuan Kerja Khusus Pelaksana Kegiatan Usaha Hulu Minyak dan Gas Bumi
            Gedung Wisma Mulia Lantai 35, Jalan Gatot Subroto 42, Jakarta 12710, Indonesia

        [3] Badan Pengelola Migas Aceh
            Jalan Stadion H. Dimurthala No.8, Kota Baru, Banda Aceh, Aceh 23125, Indonesia
"""

import numpy as np
from dataclasses import dataclass, field

############################################################################################################


@dataclass
class Oil:

    start_year: int
    end_year: int
    oil_prod: np.ndarray
    oil_price: np.ndarray
    conversion_unit: float = field(default=1.0, repr=False)

    def revenue(self):
        if len(self.oil_prod) != len(self.oil_price):
            raise ValueError(
                f'Inequal length of array: oil production: {len(self.oil_prod)}, \
                oil price: {len(self.oil_price)}'
            )

        else:
            return self.oil_prod * self.oil_price * self.conversion_unit


############################################################################################################


