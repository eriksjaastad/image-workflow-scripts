# Zip Files to Database Mapping

**Generated:** 2025-11-01  
**Purpose:** Verify mapping between zip files and existing databases before running batch predictions

## ✅ Projects WITH Databases

| Database Name         | Zip File                      | Status   | Notes                                |
| --------------------- | ----------------------------- | -------- | ------------------------------------ |
| `1011.db`             | `1011.zip`                    | ✅ Match |                                      |
| `1012.db`             | `1012.zip`                    | ✅ Match |                                      |
| `1013.db`             | `1013.zip`                    | ✅ Match |                                      |
| `1100.db`             | `1100.zip`                    | ✅ Match |                                      |
| `1101_Hailey.db`      | `1101.zip`                    | ✅ Match | Database has suffix, zip doesn't     |
| `Aiko.db`             | `Aiko_raw.zip`                | ✅ Match | DB: Aiko, Zip: Aiko_raw              |
| `Eleni.db`            | `Eleni_raw.zip`               | ✅ Match | Zip has \_raw suffix                 |
| `Kiara_Slender.db`    | `Slender Kiara.zip`           | ✅ Match | Different naming order, same project |
| `agent-1001.db`       | `agent-1001.zip`              | ✅ Match |                                      |
| `agent-1002.db`       | `agent-1002.zip`              | ✅ Match |                                      |
| `agent-1003.db`       | `agent-1003.zip`              | ✅ Match |                                      |
| `jmlimages-random.db` | `jmlimages-random.zip`        | ✅ Match |                                      |
| `mojo1.db`            | `mojo1.zip`                   | ✅ Match |                                      |
| `mojo2.db`            | `mojo2.zip`                   | ✅ Match |                                      |
| `mojo3.db`            | `mojo3.zip` (or `mojo3/` dir) | ✅ Match | Already has AI predictions           |
| `tattersail-0918.db`  | `tattersail-0918.zip`         | ✅ Match |                                      |

## ✅ Zip Files WITHOUT Databases (Will Create)

| Zip File               | Database to Create | Status              | Notes                                   |
| ---------------------- | ------------------ | ------------------- | --------------------------------------- |
| `1010.zip`             | `1010.db`          | ✅ Will create temp | AI predictions only (no user data yet)  |
| `1102.zip`             | `1102.db`          | ✅ Will create temp | AI predictions only (no user data yet)  |
| `Average Patricia.zip` | `Patricia.db`      | ✅ Will create temp | "Average Patricia" = "Patricia" project |
| `dalia.zip`            | `dalia.db`         | ✅ Will create temp | AI predictions only (no user data yet)  |
| `mixed-0919.zip`       | `mixed-0919.db`    | ✅ Will create temp | AI predictions only (no user data yet)  |

**Note:** These will get AI predictions now, user ground truth data added later via Phase 1B.

## 📋 Corrected Mapping for Script

### Projects to Process (16 total with existing databases)

```python
projects = [
    # Database Name,      Zip File
    ("1011",              "1011.zip"),
    ("1012",              "1012.zip"),
    ("1013",              "1013.zip"),
    ("1100",              "1100.zip"),
    ("1101_Hailey",       "1101.zip"),          # Note: DB has _Hailey suffix
    ("Aiko",              "Aiko_raw.zip"),      # Note: Zip has _raw suffix
    ("Eleni",             "Eleni_raw.zip"),     # Note: Zip has _raw suffix
    ("Kiara_Slender",     "Slender Kiara.zip"), # Note: Different naming order
    ("agent-1001",        "agent-1001.zip"),
    ("agent-1002",        "agent-1002.zip"),
    ("agent-1003",        "agent-1003.zip"),
    ("jmlimages-random",  "jmlimages-random.zip"),
    ("mojo1",             "mojo1.zip"),
    ("mojo2",             "mojo2.zip"),
    ("tattersail-0918",   "tattersail-0918.zip"),
    # Skip mojo3 - already has AI predictions
]
```

### Projects WITHOUT Databases (5 - need databases created first)

These zips have no existing databases, so we can't add AI predictions to them yet:

```python
# These need Phase 1A + 1B first to create initial databases
skipped_no_database = [
    ("1010",              "1010.zip"),
    ("1102",              "1102.zip"),
    ("Patricia",          "Average Patricia.zip"),
    ("dalia",             "dalia.zip"),
    ("mixed-0919",        "mixed-0919.zip"),
]
```

## 🔍 Questions to Resolve

1. **Aiko.db vs Aiko_raw.db:**

   - You have both databases
   - Only `Aiko_raw.zip` exists
   - Should we process Aiko_raw.db or Aiko.db or both?

2. **Projects without databases:**

   - Do you want to create databases for 1010, 1102, Patricia, dalia, mixed-0919?
   - Or skip them for now?

3. **mojo3:**
   - Already has AI predictions (from your earlier backfill)
   - Skip or re-process with crop proposer v3?

## 📊 Summary

**Total zip files:** 21  
**Total databases:** 17 existing (will create 5 more)  
**Will process:** 20 projects (excluding mojo3)

- 15 with existing user data
- 5 with AI predictions only (user data added later)

**All names mapped correctly:** ✅

## ✅ Recommended Action

**Process ALL 20 projects** (create temp databases for all):

**Group A - Have existing user data (15 projects):**

1. 1011, 1012, 1013
2. 1100, 1101_Hailey
3. Aiko, Eleni, Kiara_Slender
4. agent-1001, agent-1002, agent-1003
5. jmlimages-random, mojo1, mojo2
6. tattersail-0918

**Group B - AI predictions only, no user data yet (5 projects):**

1. 1010
2. 1102
3. Patricia (from "Average Patricia.zip")
4. dalia
5. mixed-0919

**Skip:**

- mojo3 (already has AI predictions from earlier backfill)

**Total:** ~182,000+ images across 20 projects
