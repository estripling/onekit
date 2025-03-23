# CHANGELOG


## v2.0.0 (2025-03-23)

### Build System

- Update versions ([#89](https://github.com/estripling/onekit/pull/89),
  [`e5b116e`](https://github.com/estripling/onekit/commit/e5b116e8ca365bfd537af15886eb3b8aba9ff173))

* build(workflows/release.yml): add spark setup

* build: update python (3.11), poetry (2.1.1), and spark (3.5.3)

- **pyproject.toml**: Include packages keyword ([#87](https://github.com/estripling/onekit/pull/87),
  [`8c8fa44`](https://github.com/estripling/onekit/commit/8c8fa44b407835420627e717a68ea6bd862474cf))

### Features

- Add dekit ([#88](https://github.com/estripling/onekit/pull/88),
  [`3c9d036`](https://github.com/estripling/onekit/commit/3c9d0365cb86c71fde8d86f6d08f4e33cb6fa3c4))

* feat(dekit): add Individual

* refactor(Individual): add __repr__

* feat(dekit): add Population

* feat(Population): add size

* test(Population): add list methods tests

* refactor(Individual.evaluate): return self

* refactor(Population): add evaluate

* feat(Population): overwrite sort

* feat(Population): add min

* feat(Population): add max

* refactor(Individual): make x a positional argument

* test(Individual): ignore immutable property inspection

* feat(numpykit): add check_random_state

* refactor(Population): add iterable type hint

* refactor(Population): add is_evaluated

* refactor(Population): update key for test_incomplete_evaluation

* test(Population): list methods - slice

* test(Individual): repr

* feat(dekit): add check_bounds

* feat(dekit): add factory for initialization strategies

* refactor: add InitializationStrategy type

* feat(Initialization): add random__standard_uniform

* refactor: rename random_real_vectors -> random__uniform

* refactor(check_random_state): check for generator first

* feat(dekit): add Mutation.rand_1

* feat(dekit): add Mutation.best_1

* feat(dekit): add crossover binomial strategies

* feat(dekit): add selection strategy

* fix(Population): is_evaluated for empty population

* feat(dekit): add bound repair strategies

* refactor(BoundsHandler): rename properties

* refactor(Initialization): rename to x_mat

* feat(dekit): add normalize and denormalize

* refactor(x_bounds): use int32

* refactor(Individual): rename property fun -> fx

* build: add dekit dependencies

* feat(Population): add generation count

* feat(Population): overwrite copy method

* test: format style

* feat(Population): add increment_generation_count

* refactor(Crossover): rename binomial methods

* refactor(Population.generation): add docstring

* refactor(are_predicates_true): type hint

* feat(dekit): add termination strategies

* feat(dekit): add differential evolution

* test: small updates

* feat: add termination message

* feat: add current_to_best

* test: parametrize mutation strategies

* feat: add rand_to_best

* refactor: consistent syntax

* feat: add termination has_reached_max_best_so_far

* refactor: add docstring for mutation strategies

* feat(Population): add shuffle

* test: shuffle

* feat(Population): add sample

* refactor: mutation strategies to use population.sample

* refactor: rename mutation functions

* refactor(Population): sample

* feat: add current_to_pbest_1

* feat: add rand_to_pbest_1

* fix: compatibility with python 3.9

* feat(dekit): add shade

* refactor(DeV3): shade 1.0

* refactor(BoundRepair): rename variable

* refactor: DeV3.get_cr_value

* feat: add shade 1.1

* docs(Dev4): add details that motivates shade 1.1

* refactor: add target to bound repair strategies

* feat(BoundRepair): add mean_target_bound

* feat(BoundRepair): add bounce

* feat(BoundRepair): add hypersphere_universe

* docs: update references

* fix(update_archive): do not exceed max_size

* feat: add population size adaption lpsr

* test: add update_archive

* test: update test_shade

* feat: add lshade

* docs: fix variant count

* refactor: rename mean_target_bound to midway

* feat: add exponential crossover

* refactor: check_random_state

* feat: add dekit.ipynb

* refactor(playbook.py): delete directories first

* build: remove .master('local[*]') from spark session for testing

* revert: remove .master('local[*]') from spark session for testing

This reverts commit f3ef8eb44c4aa422807ca63a3d1debe22881263e.

* build(workflows/test.yml): add spark setup

* build(workflows/test.yml): fix java distribution

* build(workflows/test.yml): fix spark version

* build(workflows/test.yml): update spark version

* test: do not test spark version

- Refactor sparkkit function signatures ([#90](https://github.com/estripling/onekit/pull/90),
  [`2a49323`](https://github.com/estripling/onekit/commit/2a49323b47c0b1c4b0c9ba6a54f1d3441d9a8985))

* docs(DEVELOPERS.md): update PSR link

* refactor: improve type hints

* fix(Dockerfile): workspace

* feat: refactor sparkkit function signatures

BREAKING CHANGE: function signature change

### Refactoring

- Remove curry from mathkit functions ([#92](https://github.com/estripling/onekit/pull/92),
  [`624c4c1`](https://github.com/estripling/onekit/commit/624c4c17d1b1039234cc10f7e5a4299da4a99f6f))

* refactor(digitscale): remove curry

* refactor(isdivisible): remove curry

- Rename c2.py to playbook.py ([#85](https://github.com/estripling/onekit/pull/85),
  [`bbed8f6`](https://github.com/estripling/onekit/commit/bbed8f6a6ee5a48fba95c3a4837f9c061e2226bf))

- Rename number_of_days -> num_days ([#96](https://github.com/estripling/onekit/pull/96),
  [`36bbd79`](https://github.com/estripling/onekit/commit/36bbd797800a0ceb11cab8124422d0cefbafdb52))

BREAKING CHANGE: renamed function number_of_days -> num_days

- **fetch_minima**: Remove curry ([#97](https://github.com/estripling/onekit/pull/97),
  [`e6edfa4`](https://github.com/estripling/onekit/commit/e6edfa42dc1ca0882ea79c30e0b7965b62372f84))

- **pythonkit**: Remove curry ([#94](https://github.com/estripling/onekit/pull/94),
  [`87ed704`](https://github.com/estripling/onekit/commit/87ed7041cb65bf11dfb35672d1fe54e0fe3357f7))

* refactor(date_ago): remove curry

* docs(date_ago): shorten docstring

* refactor(date_ahead): remove curry

* refactor(op): remove curry

* refactor(signif): remove curry

* refactor: improve type hints

* refactor(reduce_sets): remove curry

* refactor(pythonkit): improve type inspection

* refactor(pythonkit): rename d0 -> ref_date

BREAKING CHANGE: signature change of pythonkit functions

- **select_col_types**: Apply try/except logic ([#91](https://github.com/estripling/onekit/pull/91),
  [`1e95fe3`](https://github.com/estripling/onekit/commit/1e95fe3450d5a0777f5e691a102a18918d0904f3))

- **select_col_types**: Check if value has attr
  ([#86](https://github.com/estripling/onekit/pull/86),
  [`4c42978`](https://github.com/estripling/onekit/commit/4c429782680a08da575dc0a97bbb8942877f5fff))

- **sparkkit**: Peek ([#93](https://github.com/estripling/onekit/pull/93),
  [`df6d4f5`](https://github.com/estripling/onekit/commit/df6d4f5d3ed65e060e34ba88742cfd05d7f04332))

* feat(pythonkit): add get_shell_type

* build(pandaskit): add tabulate

* refactor(num_to_str): integrate g format

* feat(pandaskit): add display

* refactor(sparkkit): use display function in peek

BREAKING CHANGE: function signature change of sparkkit.peek

- **sparkkit**: Rename d0 -> ref_date ([#95](https://github.com/estripling/onekit/pull/95),
  [`dd9be68`](https://github.com/estripling/onekit/commit/dd9be68534b9956e070716f18bfcf54f97596ac1))

* refactor(filter_date): rename d0 -> ref_date

* refactor(sparkkit): rename d0 -> ref_date

BREAKING CHANGE: function signature change of filter_date, with_date_diff_ago, and
  with_date_diff_ahead

- **stopwatch**: Set default flush to false ([#84](https://github.com/estripling/onekit/pull/84),
  [`0f9e93f`](https://github.com/estripling/onekit/commit/0f9e93fd0ef1e5d865400e66e28b95a58df4e9a7))

### Breaking Changes

- Function signature change


## v1.5.0 (2024-07-24)

### Features

- Add precision given recall ([#83](https://github.com/estripling/onekit/pull/83),
  [`405aa2d`](https://github.com/estripling/onekit/commit/405aa2d913b3627b001b7cb80d2fc4964b0625e9))

* feat(sklearnkit): add precision_recall_values

* feat(sklearnkit): add precision_given_recall

* feat(sklearnkit): add precision_given_recall_score

* feat(sklearnkit): add precision_given_recall_summary


## v1.4.0 (2024-07-23)

### Bug Fixes

- **select_col_types**: Check for vaild data types
  ([#82](https://github.com/estripling/onekit/pull/82),
  [`36bf1e1`](https://github.com/estripling/onekit/commit/36bf1e10fe81968c44d334b65c0f17a97e90c4b8))

* fix(select_col_types): check for vaild data types

* fix(select_col_types): type hint

### Build System

- Upgrade to Python 3.9 ([#80](https://github.com/estripling/onekit/pull/80),
  [`4266907`](https://github.com/estripling/onekit/commit/42669079b7678933c32cf09281d6d9332b122644))

* build(pyproject.toml): upgrade to 3.9

* build: add scikit-learn

* build: upgrade to Python 3.9

- **c2.py**: Add help hint ([#79](https://github.com/estripling/onekit/pull/79),
  [`efdca0b`](https://github.com/estripling/onekit/commit/efdca0be5cc3b8af7994c05e68d842c797e3651f))

### Features

- Add sklearnkit ([#81](https://github.com/estripling/onekit/pull/81),
  [`a667f0d`](https://github.com/estripling/onekit/commit/a667f0da25c4bf9061496761f44d5830c4654c13))

* feat: add threshold_summary

* refactor(c2.py): use shorthand type hinting

* feat(sklearnkit): add precision_given_recall_score

* revert: "refactor(c2.py): use shorthand type hinting"

This reverts commit 3e7784ac821371dde6112076e6e25b96998f3d6b.

* test: fix casting


## v1.3.0 (2024-06-03)

### Build System

- Improve setup ([#77](https://github.com/estripling/onekit/pull/77),
  [`7b61657`](https://github.com/estripling/onekit/commit/7b616571789f0c0a962b04d5f8514c58e0377fdc))

- Replace Makefile with a Python script to run commands - Enable easy setup on host machine next to
  docker - Make sure tests pass one Windows OS

Commits ----------

* build: improve docker setup

* build: refactor clear_cache.py

* build: update poetry.lock

* build: add command.py

* build: update process_argument__create_docs

* docs: update year

* build: add remove_docs

* build(command.py): refactor main

* build(command.py): add poetry commands

* revert: "build: update poetry.lock"

This reverts commit 1e1f10cf54790358c5fee4f914a893654fcd277c.

* build: delete Makefile

* build: update commands

* fix: remove create_path

* fix: add strtobool

* build: upgrade pip

* test: fix test_highlight_string_differences

* test: fix test_lazy_read_lines

* test: fix TestTimestamp.test_default_call

* test: fix TestStopwatch no regex match

* test: fix test_vizkit.py

* refactor: run_remove_docs

* build: add get_last_commit

* build: rename command.py to c2.py

- Update poetry ([#74](https://github.com/estripling/onekit/pull/74),
  [`bae0e62`](https://github.com/estripling/onekit/commit/bae0e6226e08c36d4a245986fe97a40a620c5ba3))

### Features

- Add bool_to_str ([#78](https://github.com/estripling/onekit/pull/78),
  [`fadd710`](https://github.com/estripling/onekit/commit/fadd710a52ba0f18942796a563017912bef5f595))

* refactor(bool_to_int): use select_col_types

* feat: add bool_to_str

- Add select_col_types ([#76](https://github.com/estripling/onekit/pull/76),
  [`0720987`](https://github.com/estripling/onekit/commit/0720987349a5c70bc99d3b116796d09905f00024))

- Extend digitscale ([#75](https://github.com/estripling/onekit/pull/75),
  [`80771d3`](https://github.com/estripling/onekit/commit/80771d3b865bc1a3d26901726209bd7009b2d25a))

* feat(digitscale): add kind

* feat(digitscale): add linear kind

* style: be consistent


## v1.2.0 (2024-05-10)

### Build System

- Improve remove-local-branches ([#72](https://github.com/estripling/onekit/pull/72),
  [`0dcd997`](https://github.com/estripling/onekit/commit/0dcd9978cd76e1e9d82efa957fa5830b788c8154))

- Update github workflows ([#73](https://github.com/estripling/onekit/pull/73),
  [`efb2f19`](https://github.com/estripling/onekit/commit/efb2f19a0dd71e724b1526bc3b214729bd373db7))

* build: update github workflows

* build: move time-machine dep to testing group

### Code Style

- Change wording ([#71](https://github.com/estripling/onekit/pull/71),
  [`639d914`](https://github.com/estripling/onekit/commit/639d914225f2a751fbc368acd438a1c6eee0c795))

### Documentation

- Update onekit logo ([#70](https://github.com/estripling/onekit/pull/70),
  [`dce595c`](https://github.com/estripling/onekit/commit/dce595c1a148af48d0b500d532367aee20571750))

### Features

- Add digitscale ([#69](https://github.com/estripling/onekit/pull/69),
  [`4f9c908`](https://github.com/estripling/onekit/commit/4f9c908aa29d20d3b04c18127593b1d86f9a8b18))

* feat(mathkit): add digitscale

* feat(mathkit): add sign

* feat(numpykit): add digitscale

* feat(sparkkit): add with_digitscale

* docs: add mathkit.ipynb


## v1.1.1 (2024-05-06)

### Bug Fixes

- Dependencies ([#68](https://github.com/estripling/onekit/pull/68),
  [`cb34fc7`](https://github.com/estripling/onekit/commit/cb34fc7cc627e90a5add59404d02e2236f716cec))

* build: add pytz

* fix: dependencies

### Refactoring

- **filter_date**: N=inf for no lower bound ([#66](https://github.com/estripling/onekit/pull/66),
  [`46e3606`](https://github.com/estripling/onekit/commit/46e3606739f91d4eee995cc56a57b576a4bf04b8))

### Testing

- Refactor spark session ([#67](https://github.com/estripling/onekit/pull/67),
  [`dc34fe7`](https://github.com/estripling/onekit/commit/dc34fe7d37de8967f034686924472e89bf8321ae))


## v1.1.0 (2024-04-26)

### Features

- Add timestamp ([#64](https://github.com/estripling/onekit/pull/64),
  [`67e657d`](https://github.com/estripling/onekit/commit/67e657d19ea5f20afaafe276cfd159cbe8f81ee4))

* feat(pythonkit): add timestamp

* docs(LICENSE): update year

* refactor(timestamp): no positional arguments

- **stopwatch**: Add timezone ([#65](https://github.com/estripling/onekit/pull/65),
  [`d261904`](https://github.com/estripling/onekit/commit/d2619049e437f6cca9032d82046a6b7247134f90))

* docs: use consistent style for optional parameters

* refactor(archive_files): add timezone

* feat(stopwatch): add timezone

### Refactoring

- **filter_date**: Use with_date_diff_ago ([#63](https://github.com/estripling/onekit/pull/63),
  [`b0bd1ec`](https://github.com/estripling/onekit/commit/b0bd1ecd4b0e6d8c51e12854d6d10601dc1b8882))


## v1.0.0 (2024-04-01)

### Code Style

- Change type hint for star parameters ([#53](https://github.com/estripling/onekit/pull/53),
  [`840bc66`](https://github.com/estripling/onekit/commit/840bc66e371be24b254fcc87b1836e0c57b968aa))

### Features

- Add all_col and any_col ([#52](https://github.com/estripling/onekit/pull/52),
  [`fb0c07a`](https://github.com/estripling/onekit/commit/fb0c07a55e30e1db6d598b7a47cb0997cfa4a47b))

* feat: add all_col

* feat: add any_col

- Add bool_to_int ([#58](https://github.com/estripling/onekit/pull/58),
  [`bbde5aa`](https://github.com/estripling/onekit/commit/bbde5aa04fadb42de40e2d46d85e0300180d70a0))

* feat: add bool_to_int

* fix(peek): use bool_to_int before converting to pandas

- Add date diff ([#55](https://github.com/estripling/onekit/pull/55),
  [`2504dcd`](https://github.com/estripling/onekit/commit/2504dcdf86a77454208c1bff4647ed7c1bc20b9c))

* docs(str_to_col): update docstring

* feat: add with_date_diff

- Add with_increasing_id ([#62](https://github.com/estripling/onekit/pull/62),
  [`1dd4c19`](https://github.com/estripling/onekit/commit/1dd4c19512d8135f67bc4b18d00d3911ef9767ac))

* feat: add with_increasing_id

* test: fix nondeterministic test

- Datecount -> date_count_backward, date_count_forward
  ([#59](https://github.com/estripling/onekit/pull/59),
  [`fff3c77`](https://github.com/estripling/onekit/commit/fff3c77ef58c1ba7e96f430e6b78d48d15aebf20))

BREAKING CHANGE: split datecount into date_count_backward and date_count_forward

- Improve date functionality ([#54](https://github.com/estripling/onekit/pull/54),
  [`72fba47`](https://github.com/estripling/onekit/commit/72fba4706626ff8b18e208e86271e9595ebe1962))

* feat: add filter_date

* refactor: rename day -> d

* refactor: rename daycount -> datecount

* refactor: rename n_days -> number_of_days

* test(relative_date): rename days -> dates

* feat: add date_ahead

* feat: add date_ago

* fix: remove relative_date

* refactor: change order

BREAKING CHANGE: renamed date functions

- Split with_date_diff into with_date_diff_ago and with_date_diff_ahead
  ([#57](https://github.com/estripling/onekit/pull/57),
  [`1175784`](https://github.com/estripling/onekit/commit/11757848773a9378e89bcc9bb5d3c5b535105a33))

### Refactoring

- Rename daterange -> date_range ([#60](https://github.com/estripling/onekit/pull/60),
  [`547d5ac`](https://github.com/estripling/onekit/commit/547d5ac9a040843a945e76cb172da503fae7f3c8))

BREAKING CHANGE: rename daterange -> date_range

- Rename sk.daterange -> sk.date_range ([#61](https://github.com/estripling/onekit/pull/61),
  [`c1b8661`](https://github.com/estripling/onekit/commit/c1b8661025fbf7bab6be4ea682c436dff330ed98))

* refactor: rename sk.daterange -> sk.date_range

* refactor(with_weekday): add type hint for determine_weekday

BREAKING CHANGE: rename sk.daterange -> sk.date_range

### Testing

- Refactor to have shorter tests ([#56](https://github.com/estripling/onekit/pull/56),
  [`7aa966d`](https://github.com/estripling/onekit/commit/7aa966d409498c03b2b85b2552d8b6952105fd1b))

* test: refactor test_with_endofweek_date

* test: refactor test_with_index

* test: refactor test_with_startofweek_date

* test: refactor test_with_weekday

* test: refactor test_filter_date

### Breaking Changes

- Rename sk.daterange -> sk.date_range


## v0.16.0 (2024-03-03)

### Features

- Add check column functionality ([#51](https://github.com/estripling/onekit/pull/51),
  [`639f379`](https://github.com/estripling/onekit/commit/639f379429ac0cfaeaea821190649ac7782aa8ce))

* feat: add check_column_present

* feat: add has_column

### Refactoring

- Rename check functions ([#50](https://github.com/estripling/onekit/pull/50),
  [`0bd7103`](https://github.com/estripling/onekit/commit/0bd7103ff176e5c3645a8f3596ffebaa779d1831))

* refactor: rename check_dataframe_equal -> assert_dataframe_equal

* refactor: rename check_row_count_equal -> assert_row_count_equal

* refactor: rename check_row_equal -> assert_row_equal

* refactor: rename check_schema_equal -> assert_schema_equal

* refactor(example.ipynb): update notebook

* build: add pre-commit hook


## v0.15.0 (2024-01-29)

### Features

- **pandaskit**: Add cvf ([#49](https://github.com/estripling/onekit/pull/49),
  [`a8c9cb6`](https://github.com/estripling/onekit/commit/a8c9cb66e87f1ff3ef968a6604be96c110f23151))


## v0.14.0 (2024-01-24)

### Features

- Improve pandaskit.profile ([#48](https://github.com/estripling/onekit/pull/48),
  [`f10bf9a`](https://github.com/estripling/onekit/commit/f10bf9a0acf4c981dcd1e5263a2a2afbee28de81))

* refactor(pandaskit.profile): no special characters

* test(pandaskit.profile): add isnull and notull query

* feat(numpykit): add quantile argument

* test: rename function

* refactor(pandaskit.profile): use basic_info_df variable

* feat(pandaskit.profile): add sum


## v0.13.0 (2023-12-22)

### Features

- Add numpykit ([#47](https://github.com/estripling/onekit/pull/47),
  [`d580b8f`](https://github.com/estripling/onekit/commit/d580b8f3167731c028dc65c62b7b76f46e7d259e))

* refactor(check_vector): move optfunckit -> numpykit

* feat(numpykit): add stderr

### Refactoring

- Move math functions to mathkit ([#46](https://github.com/estripling/onekit/pull/46),
  [`c4a5bd8`](https://github.com/estripling/onekit/commit/c4a5bd8c83986b626d6c7d2b01d6ee920b746263))

* refactor(collatz): move pythonkit -> mathkit

* refactor(fibonacci): move pythonkit -> mathkit

* refactor: move isdivisible iseven isodd to mathkit

- Small updates ([#45](https://github.com/estripling/onekit/pull/45),
  [`c6f02d7`](https://github.com/estripling/onekit/commit/c6f02d701965b6c34ae50392f5b2fdd5a573d654))

* refactor(XyzPoints): adjust type hints

* docs(peaks): add reference

* docs(optfunckit): add ai-roomi reference

* docs(concat_strings): add docstring example


## v0.12.0 (2023-12-21)

### Features

- Add fbench functions ([#43](https://github.com/estripling/onekit/pull/43),
  [`173052f`](https://github.com/estripling/onekit/commit/173052f102696f8cb2f0652c7e8e19875f43556e))

* feat: add test functions for minimization

* feat(optfunckit): add negate

* feat(optfunckit): add bump

* refactor(sinc): negate

* style(optfunckit): correct indentation

* docs(peaks): add reference in docstring

* docs(negate): add maximization example

- Add vizkit ([#44](https://github.com/estripling/onekit/pull/44),
  [`5f5e824`](https://github.com/estripling/onekit/commit/5f5e82466d69673c21c3e436787f0005de21bc3b))

* build: add matplotlib dependency

* feat(pythonkit): add op

* feat: add vizkit

### Refactoring

- Error messages ([#42](https://github.com/estripling/onekit/pull/42),
  [`e7e353e`](https://github.com/estripling/onekit/commit/e7e353e234b8ebd32ce2b036e51122033e5c8837))


## v0.11.0 (2023-12-11)

### Features

- Add pandaskit ([#39](https://github.com/estripling/onekit/pull/39),
  [`60e9a09`](https://github.com/estripling/onekit/commit/60e9a094cc38e05566609ed31d6d87934bf27663))

* feat(pandaskit): add union

* feat(pandaskit): add join

* test: refactor pandas test_union

* feat(pandaskit): add profile

- Add spark functions ([#41](https://github.com/estripling/onekit/pull/41),
  [`d421ac4`](https://github.com/estripling/onekit/commit/d421ac40170bf0c2d9222d1cc7cab618a73d7678))

* feat(sparkkit): add count_nulls

* feat(sparkkit): add join

* feat(sparkkit): add add_prefix

* refactor(count_nulls): add asterisk in signature

* feat(sparkkit): add add_suffix

* refactor: curry functions

* feat(sparkkit): add daterange

* refactor(daycount): rename start to d0

* refactor(daterange): rename parameters

* feat(sparkkit): add with_index

* style(with_index): positional parameter

* refactor: use spark transform func

* feat(sparkkit): add with_weekday

* feat(sparkkit): add with_endofweek_date

* style(with_index): rephrase docstring header

* feat(sparkkit): add with_startofweek_date

* refactor: use transform pattern

* build(Makefile): add commands for running slow tests

* feat(sparkkit): add validation functions

* feat(sparkkit): add evaluation functions

### Refactoring

- Rename num variables + type hints ([#40](https://github.com/estripling/onekit/pull/40),
  [`6269e3a`](https://github.com/estripling/onekit/commit/6269e3a38ab07b9cb685cec769b65e3bdb8036fa))

* refactor: int values as n

* test: add type hints


## v0.10.1 (2023-11-22)

### Bug Fixes

- **README.md**: Disclaimer section
  ([`9428fee`](https://github.com/estripling/onekit/commit/9428feef002ba4648f1c3f63f2de6f5ad7b1fd1d))

### Documentation

- Update onekit author and description ([#38](https://github.com/estripling/onekit/pull/38),
  [`62947fb`](https://github.com/estripling/onekit/commit/62947fb6a8be9711fc55e31cac676697e4326022))

* docs: update license

* build: update pyproject.toml

* docs: update README.md

* docs: update author


## v0.10.0 (2023-11-22)

### Documentation

- **sparkkit.peek**: Add missing shape argument
  ([#36](https://github.com/estripling/onekit/pull/36),
  [`7810dc3`](https://github.com/estripling/onekit/commit/7810dc3e954483ae482c8cda59289513fad6a88b))

* docs(sparkkit.peek): add missing shape argument

* refactor: DfIdentityFunction -> DFIdentityFunc

### Features

- **sparkkit**: Add cvf ([#37](https://github.com/estripling/onekit/pull/37),
  [`b989f31`](https://github.com/estripling/onekit/commit/b989f31bb4a9fa9a0e40eed24852cf8b7a6b1335))

* feat(sparkkit): add str_to_col

* refactor: DFIdentityFunc -> SparkDFIdentityFunc

* style(peek): remove docstring of inner function

* test(str_to_col): no parametrize

* feat(sparkkit): add cvf


## v0.9.0 (2023-11-20)

### Features

- **sparkkit**: Add peek ([#35](https://github.com/estripling/onekit/pull/35),
  [`69b08e9`](https://github.com/estripling/onekit/commit/69b08e90e6399639bf013414a0b6d45d1c008393))

### Refactoring

- Rename modules ([#33](https://github.com/estripling/onekit/pull/33),
  [`5dfd157`](https://github.com/estripling/onekit/commit/5dfd157ae6c4ae7e89afe58e121f7dde4d2f719e))

* refactor: pytlz -> pythonkit

* refactor: sparktlz -> sparkkit

* build(pyproject.toml): rename sparktlz -> sparkkit

* refactor: pdtlz -> pandaskit

BREAKING CHANGE: rename modules to have kit suffix

- **signif**: Curry function ([#34](https://github.com/estripling/onekit/pull/34),
  [`6880f9c`](https://github.com/estripling/onekit/commit/6880f9c297f1e178612b72187528a289ffafc25a))


## v0.8.0 (2023-11-17)

### Documentation

- Add notebook examples ([#32](https://github.com/estripling/onekit/pull/32),
  [`59c972b`](https://github.com/estripling/onekit/commit/59c972b9507c87bef0fe37012065b69b5632c810))

* docs: add examples

* docs(example.ipynb): add highlight_string_differences

* docs(example.ipynb): add stopwatch

* refactor: example.ipynb -> examples.ipynb

### Features

- Migrate bumbag io functions ([#27](https://github.com/estripling/onekit/pull/27),
  [`4fd81a3`](https://github.com/estripling/onekit/commit/4fd81a37296df9e7c71584ad2155e45a6b9bb6b9))

* feat(pytlz): add lazy_read_lines

* feat(pytlz): add prompt_yes_no

* feat(pytlz): add archive_files

- Migrate bumbag time functions ([#29](https://github.com/estripling/onekit/pull/29),
  [`692542b`](https://github.com/estripling/onekit/commit/692542beec78adc4ff272472b305451408a6e94e))

* feat(pytlz): add humantime

* style: fix minor format

* feat(pytlz): add stopwatch

* feat(pytlz): add str_to_date

* feat(pytlz): add weekday

* refactor: use from toolz import curried

* test(filter_regex): remove text

* feat(pytlz): add daycount

* feat(pytlz): add daterange

* feat(pytlz): add last_date_of_month

* feat(pytlz): add n_days

* feat(pytlz): add relative_date

### Refactoring

- Isdivisibleby -> isdivisible ([#31](https://github.com/estripling/onekit/pull/31),
  [`bb9af5a`](https://github.com/estripling/onekit/commit/bb9af5a4893c367f36acd85b34c92751242bc8b6))

- **relative_date**: Change signature ([#30](https://github.com/estripling/onekit/pull/30),
  [`049ab7d`](https://github.com/estripling/onekit/commit/049ab7d9745f2688e383dc2607202f11908aca42))

### Testing

- Add itertools ([#28](https://github.com/estripling/onekit/pull/28),
  [`b83469d`](https://github.com/estripling/onekit/commit/b83469d1a05c139e8ff6f941119e805ba4dca4a1))


## v0.7.0 (2023-11-17)

### Documentation

- Update docstrings ([#24](https://github.com/estripling/onekit/pull/24),
  [`1585554`](https://github.com/estripling/onekit/commit/15855544eb2b85abe55ff34908028e9c10d45a54))

* docs(reduce_sets): update docstring example

* docs(source_code): update docstring

### Features

- Migrate bumbag string functions ([#26](https://github.com/estripling/onekit/pull/26),
  [`95b8e7d`](https://github.com/estripling/onekit/commit/95b8e7dc4b405c58b49a021fea26f2ac20a87f07))

* feat(pytlz): add concat_strings

* feat(pytlz): add create_path

* feat(pytlz): add filter_regex

* refactor: use iterable instead of sequence

* feat(pytlz): add map_regex

* feat(pytlz): add headline

* feat(pytlz): add remove_punctuation

* feat(pytlz): add highlight_string_differences

- **pytlz**: Add are_predicates_true ([#25](https://github.com/estripling/onekit/pull/25),
  [`de96017`](https://github.com/estripling/onekit/commit/de960173809febc8947b80e2a7735ae140574fd7))

- Apply DRY principle: replace all_predicates_true and any_predicates_true


## v0.6.0 (2023-11-15)

### Documentation

- **pytlz**: Rephrase docstring of bool functions
  ([#20](https://github.com/estripling/onekit/pull/20),
  [`fa8a1df`](https://github.com/estripling/onekit/commit/fa8a1df96aaf72e77f9462c0ac4bd161784d4ac0))

### Features

- Migrate bumbag random functions ([#22](https://github.com/estripling/onekit/pull/22),
  [`8906f15`](https://github.com/estripling/onekit/commit/8906f15047a57970b88eb61aef52e20a6210a60b))

* feat(pytlz): add check_random_state

* feat(pytlz): add coinflip

* test: add raises checks in else clause

* docs(coinflip): add docstring example

* docs(coinflip): add docstring example with biased coin

- **pytlz**: Add collatz and fibonacci ([#21](https://github.com/estripling/onekit/pull/21),
  [`0881625`](https://github.com/estripling/onekit/commit/088162506219ef5a95acb3dbde81dd108d686071))

* feat(pytlz): add collatz

* feat(pytlz): add fibonacci

* style(collatz): update references in docstring

### Refactoring

- Curry functions only where necessary ([#23](https://github.com/estripling/onekit/pull/23),
  [`acbbcea`](https://github.com/estripling/onekit/commit/acbbcea2f31f063469cb140ea70f18d63cbea099))

* refactor(extend_range): replace curry with partial

* docs(isdivisibleby): indicate function is curried

* docs(reduce_sets): indicate function is curried

* refactor(signif): replace curry with partial


## v0.5.0 (2023-11-14)

### Build System

- **pyproject.toml**: Update classifiers ([#14](https://github.com/estripling/onekit/pull/14),
  [`6bdc390`](https://github.com/estripling/onekit/commit/6bdc3906c552caa12336aafa7f8a4f4020022e40))

### Code Style

- Update docs ([#11](https://github.com/estripling/onekit/pull/11),
  [`2527c1c`](https://github.com/estripling/onekit/commit/2527c1c6bc5bdfdc87b2bbc913bf2585e782daaf))

* docs: update module docstring

* refactor(Makefile): add missing phony

* docs(pytlz.flatten): update type hinting

* docs(sparktlz.union): update type hinting

* refactor: rename SparkDataFrame -> SparkDF

### Continuous Integration

- **release.yml**: Use release token ([#12](https://github.com/estripling/onekit/pull/12),
  [`f2ca10a`](https://github.com/estripling/onekit/commit/f2ca10a65ca2e3828685148a2a96f3934afe51cd))

### Documentation

- Update developer guide ([#17](https://github.com/estripling/onekit/pull/17),
  [`3d141b5`](https://github.com/estripling/onekit/commit/3d141b528a2823c555e6b8c364707f3291e5856f))

- **README.md**: Remove example ([#13](https://github.com/estripling/onekit/pull/13),
  [`9f510ac`](https://github.com/estripling/onekit/commit/9f510ace488f6a77b1b482432afc09204e82e2d2))

### Features

- Add bumbag core functions ([#18](https://github.com/estripling/onekit/pull/18),
  [`50badd8`](https://github.com/estripling/onekit/commit/50badd8d105adc8c9d7acc94f70ef0fcffaa8af1))

* build: add toolz

* test: add toolz

* feat(pytlz): add isdivisibleby

* feat(pytlz): add iseven

* feat(pytlz): add isodd

* feat(pytlz): add all_predicate_true

* feat(pytlz): add any_predicate_true

* test: use leaner syntax

* test: mark Spark tests as slow

* test: use test class for toolz

* build: add pytest-skip-slow

* test(TestSparkToolz): refactor assert_dataframe_equal

* feat(pytlz): add extend_range

* feat(pytlz): add func_name

* feat(pytlz): add source_code

* feat(pytlz): add signif

* feat(pytlz): add reduce_sets

* refactor(num_to_str): consistent type hinting

* feat(pytlz): add contrast_sets

* docs(all_predicate_true): add type call to show it is curried

* docs(any_predicate_true): add type call to show it is curried

* docs(extend_range): add type call to show it is curried

* docs(isdivisibleby): add type call to show it is curried

* docs(signif): add type call to show it is curried

### Refactoring

- Predicate functions ([#19](https://github.com/estripling/onekit/pull/19),
  [`2a9f699`](https://github.com/estripling/onekit/commit/2a9f699d6c78b8f91a2a762f74ae7935a75c8ce2))

* refactor(all_predicate_true): use inner function

* refactor(any_predicate_true): use inner function

* docs(isdivisibleby): update docstring for consistency

### Testing

- Ignore Spark doctest ([#16](https://github.com/estripling/onekit/pull/16),
  [`e682a09`](https://github.com/estripling/onekit/commit/e682a09f86a595b1549356322dc725e99092ff84))


## v0.4.0 (2023-11-13)

### Build System

- Add pandas ([#9](https://github.com/estripling/onekit/pull/9),
  [`a5185ba`](https://github.com/estripling/onekit/commit/a5185baf15be09815fed92c34a70191ebdc86a91))

### Documentation

- Add example ([#10](https://github.com/estripling/onekit/pull/10),
  [`9af41f3`](https://github.com/estripling/onekit/commit/9af41f347198d43d42def455cf6e2076992fe603))

* docs(example.ipynb): add pytlz.flatten and sparktlz.union

* style(sparktlz.union): rearrange docsting imports

* docs(README.md): add example

- Import pytlz directly ([#6](https://github.com/estripling/onekit/pull/6),
  [`e777ec4`](https://github.com/estripling/onekit/commit/e777ec440be757da0cd7a896563580f8d343667a))

* refactor: import pytlz directly

* docs: no import onekit as ok

* docs(README.md): rename example usage to examples

* docs: add module description

### Features

- Add sparktlz ([#5](https://github.com/estripling/onekit/pull/5),
  [`443dc33`](https://github.com/estripling/onekit/commit/443dc338543f301da19d4a656e65bb7632949363))

* build: add pyspark

* tests: set up Spark session

- **pytlz**: Add flatten ([#7](https://github.com/estripling/onekit/pull/7),
  [`7629260`](https://github.com/estripling/onekit/commit/7629260f14a0ce7af3afa78e624753f03efd55e1))

* tests(date_to_str): correct test name

* feat(pytlz): add flatten

- **sparktlz**: Add union ([#8](https://github.com/estripling/onekit/pull/8),
  [`7680762`](https://github.com/estripling/onekit/commit/76807626f3c64755171654d43749eb9c875c9d8c))


## v0.3.0 (2023-11-09)

### Features

- Add date_to_str ([#3](https://github.com/estripling/onekit/pull/3),
  [`1068a65`](https://github.com/estripling/onekit/commit/1068a65df00a9c89997f3fe8c216e4e03f818e59))

* num_to_str: rephrase docstring

* pytlz: add date_to_str

### Refactoring

- Import onekit as ok ([#4](https://github.com/estripling/onekit/pull/4),
  [`aa077f8`](https://github.com/estripling/onekit/commit/aa077f818ed942d6f7742ee79d08e435af893a19))

- Num_to_str ([#2](https://github.com/estripling/onekit/pull/2),
  [`2447205`](https://github.com/estripling/onekit/commit/2447205871da761d1ac321475fac4bea1ee6107e))

* num_to_str: improve type hinting

* changelog.md: remove bullet points


## v0.2.0 (2023-11-09)

### Continuous Integration

- Github release before PyPI
  ([`1199208`](https://github.com/estripling/onekit/commit/11992089d2396ef24f81bfa2c1e5624750c8b3b5))

### Features

- **pytlz**: Add num_to_str ([#1](https://github.com/estripling/onekit/pull/1),
  [`cc130e6`](https://github.com/estripling/onekit/commit/cc130e6850d9a8a325067a0935ccceada27b0e40))

* feat(pytlz): add num_to_str

* docs: show pytlz module


## v0.1.0 (2023-11-09)

### Features

- Add repository setup
  ([`2603f02`](https://github.com/estripling/onekit/commit/2603f02007d10a8b932a0510495df89a9c64635b))
