"""
Benchmark test pitanja za evaluaciju sistema.

Ovaj fajl sadrži test pitanja sa očekivanim odgovorima
za različite domene (projekti, HR, itd.)
"""

# =============================================================================
# PROJEKTNA PITANJA
# =============================================================================

PROJECT_BENCHMARK_TESTS = [
    {
        "id": "T01",
        "question": "pasars lot 2 vozilo 5",
        "expected": {
            "intent": "project_status",
            "project": "PASARS",
            "lot": 2,
            "vehicle": 5,
            "completion_percent": 25,
        },
        "description": "Direktno pitanje - latinica, lot 2, vozilo 5",
    },
    {
        "id": "T02",
        "question": "ПАСАРС лот 1 возило 1",
        "expected": {
            "intent": "project_status",
            "project": "PASARS",
            "lot": 1,
            "vehicle": 1,
            "completion_percent": 100,
        },
        "description": "Ćirilica - završeno vozilo (100%)",
    },
    {
        "id": "T03",
        "question": "koliko je završeno vozilo 3 u MRAP projektu lot 1",
        "expected": {
            "intent": "project_status",
            "project": "MRAP",
            "lot": 1,
            "vehicle": 3,
            "completion_percent": 80,
        },
        "description": "Pitanje o procentu završenosti - MRAP",
    },
    {
        "id": "T04",
        "question": "status NTV vozilo 6 lot 2",
        "expected": {
            "intent": "project_status",
            "project": "NTV",
            "lot": 2,
            "vehicle": 6,
            "completion_percent": 0,
        },
        "description": "Vozilo koje nije započeto (0%)",
    },
    {
        "id": "T05",
        "question": "МРАП лот 3 возило 7",
        "expected": {
            "intent": "project_status",
            "project": "MRAP",
            "lot": 3,
            "vehicle": 7,
            "completion_percent": 20,
        },
        "description": "Ćirilica - MRAP lot 3",
    },
    {
        "id": "T06",
        "question": "ntv lot 1 vozilo 2 procenat",
        "expected": {
            "intent": "project_status",
            "project": "NTV",
            "lot": 1,
            "vehicle": 2,
            "completion_percent": 90,
        },
        "description": "Mala slova + reč 'procenat'",
    },
    {
        "id": "T07",
        "question": "da li je vozilo 4 u pasars lot 2 završeno",
        "expected": {
            "intent": "project_status",
            "project": "PASARS",
            "lot": 2,
            "vehicle": 4,
            "completion_percent": 50,
        },
        "description": "Da/ne pitanje o završenosti",
    },
    {
        "id": "T08",
        "question": "mrap vozilo broj 9 lot 3",
        "expected": {
            "intent": "project_status",
            "project": "MRAP",
            "lot": 3,
            "vehicle": 9,
            "completion_percent": 0,
        },
        "description": "MRAP vozilo koje nije započeto",
    },
    {
        "id": "T09",
        "question": "НТВ возило 4",
        "expected": {
            "intent": "project_status",
            "project": "NTV",
            "lot": 2,
            "vehicle": 4,
            "completion_percent": 50,
        },
        "description": "Ćirilica bez eksplicitnog lota",
    },
    {
        "id": "T10",
        "question": "pasars lot 1 vozilo 2 status realizacije",
        "expected": {
            "intent": "project_status",
            "project": "PASARS",
            "lot": 1,
            "vehicle": 2,
            "completion_percent": 90,
        },
        "description": "Duže pitanje sa 'status realizacije'",
    },
]


# =============================================================================
# HR PITANJA
# =============================================================================

HR_BENCHMARK_TESTS = [
    {
        "id": "HR01",
        "question": "kada ističe ugovor Lazi Lazareviću",
        "expected": {
            "intent": "hr",
            "employee_name": "laza lazarevic",
            "employment_status": "Одређено",
            "contract_end_date": "01.07.2026",
            "department": "Служба одржавања и енергетике",
        },
        "description": "Pitanje o isteku ugovora - latinica",
    },
    {
        "id": "HR02",
        "question": "ko ima opomenu pred otkaz",
        "expected": {
            "intent": "hr",
            "employee_name": "laza lazarevic",
            "has_warning": True,
            "warning_keywords": ["опомена пред отказ", "opomena pred otkaz"],
        },
        "description": "Pitanje o opomeni pred otkaz",
    },
    {
        "id": "HR03",
        "question": "ko je sistem administrator",
        "expected": {
            "intent": "hr",
            "employee_name": "dusan savic",
            "position": "Систем администратор",
            "department": "IT служба",
        },
        "description": "Pitanje o radnom mestu - sistem admin",
    },
    {
        "id": "HR04",
        "question": "ko radi u službi montaže",
        "expected": {
            "intent": "hr",
            "department": "Служба монтаже",
            "expected_employees": ["milan petrovic", "vladimir popovic", "natasa lukic"],
        },
        "description": "Pitanje o zaposlenima u sektoru",
    },
    {
        "id": "HR05",
        "question": "Душан Савић у ком sektoru radi",
        "expected": {
            "intent": "hr",
            "employee_name": "dusan savic",
            "department": "IT служба",
            "position": "Систем администратор",
        },
        "description": "Ćirilično ime + latinično pitanje",
    },
    {
        "id": "HR06",
        "question": "ko je na neodređeno",
        "expected": {
            "intent": "hr",
            "employment_status": "Неодређено",
            "expected_employees": ["jelena nikolic", "ana stankovic", "marija pavlovic", "dusan savic"],
        },
        "description": "Pitanje o statusu zaposlenja - neodređeno",
    },
    {
        "id": "HR07",
        "question": "šta je sledeća aktivnost za Jelenu Nikolić",
        "expected": {
            "intent": "hr",
            "employee_name": "jelena nikolic",
            "next_action": "Редовна годишња процена",
        },
        "description": "Pitanje o sledećoj aktivnosti",
    },
    {
        "id": "HR08",
        "question": "Марија Павловић pozicija",
        "expected": {
            "intent": "hr",
            "employee_name": "marija pavlovic",
            "position": "HR специјалиста",
            "department": "Људски ресурси",
        },
        "description": "Ćirilica - pitanje o poziciji", 
    },
    {
        "id": "HR09",
        "question": "ko radi u inženjeringu",
        "expected": {
            "intent": "hr",
            "department": "Инжењеринг",
            "expected_employees": ["ivan kostic", "goran antic"],
        },
        "description": "Pitanje o sektoru - inženjering",
    },
    {
        "id": "HR10",
        "question": "kada je počeo da radi Goran Antić",
        "expected": {
            "intent": "hr",
            "employee_name": "goran antic",
            "contract_start_date": "25.05.2017",
            "employment_status": "Неодређено",
        },
        "description": "Pitanje o datumu početka rada",
    },
    {
        "id": "HR11",
        "question": "ko radi u finansijama",
        "expected": {
            "intent": "hr",
            "department": "Финансије",
            "expected_employees": ["ana stankovic"],
        },
        "description": "Pitanje o sektoru finansija",
    },
    {
        "id": "HR12",
        "question": "ko je na probnom radu",
        "expected": {
            "intent": "hr",
            "expected_employees": ["katarina dordevic"],
            "warning_keywords": ["пробни рад", "probni rad"],
        },
        "description": "Pitanje o zaposlenima na probnom radu",
    },
    {
        "id": "HR13",
        "question": "ko ima disciplinsku napomenu",
        "expected": {
            "intent": "hr",
            "expected_employees": ["ivana matic"],
            "warning_keywords": ["дисциплинска напомена", "disciplinska"],
        },
        "description": "Pitanje o disciplinskim merama",
    },
    {
        "id": "HR14",
        "question": "ko je vođa smene",
        "expected": {
            "intent": "hr",
            "employee_name": "vladimir popovic",
            "position": "Вођа смене",
            "department": "Служба монтаже",
        },
        "description": "Pitanje o specifičnoj poziciji - vođa smene",
    },
    {
        "id": "HR15",
        "question": "Софија Миленковић kada joj ističe ugovor",
        "expected": {
            "intent": "hr",
            "employee_name": "sofija milenkovic",
            "contract_end_date": "31.10.2026",
            "department": "Набавка",
        },
        "description": "Ćirilica ime + pitanje o isteku",
    },
    {
        "id": "HR16",
        "question": "ko radi u pravnom sektoru",
        "expected": {
            "intent": "hr",
            "department": "Правни сектор",
            "expected_employees": ["tamara radovanovic"],
        },
        "description": "Pitanje o pravnom sektoru",
    },
    {
        "id": "HR17",
        "question": "ko je zadužen za bezbednost i zdravlje na radu",
        "expected": {
            "intent": "hr",
            "employee_name": "aleksandar vucicevic",
            "position": "Инжењер БЗР",
            "department": "Безбедност и здравље на раду",
        },
        "description": "Pitanje o BZR sektoru",
    },
    {
        "id": "HR18",
        "question": "kome ističe ugovor u decembru 2026",
        "expected": {
            "intent": "hr",
            "expected_employees": ["marko jovanovic", "tamara radovanovic", "natasa lukic"],
            "contract_end_date": "31.12.2026",
        },
        "description": "Pitanje o isteku ugovora po mesecu",
    },
    {
        "id": "HR19",
        "question": "ko ima problem sa izostancima",
        "expected": {
            "intent": "hr",
            "employee_name": "petar stevanovic",
            "warning_keywords": ["изостанака", "izostanak", "izostanci"],
        },
        "description": "Pitanje o izostancima",
    },
    {
        "id": "HR20",
        "question": "ko radi u korisničkoj podršci i kada mu ističe ugovor",
        "expected": {
            "intent": "hr",
            "employee_name": "sara zivkovic",
            "department": "Корисничка подршка",
            "contract_end_date": "30.11.2026",
            "position": "Специјалиста подршке",
        },
        "description": "Kombinovano pitanje - sektor + istek",
    },
]
