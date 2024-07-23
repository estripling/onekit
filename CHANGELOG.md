# CHANGELOG

## v1.4.0 (2024-07-23)

### Build

* build: upgrade to Python 3.9 (#80)

* build(pyproject.toml): upgrade to 3.9

* build: add scikit-learn

* build: upgrade to Python 3.9 ([`4266907`](https://github.com/estripling/onekit/commit/42669079b7678933c32cf09281d6d9332b122644))

* build(c2.py): add help hint (#79) ([`efdca0b`](https://github.com/estripling/onekit/commit/efdca0be5cc3b8af7994c05e68d842c797e3651f))

### Feature

* feat: add sklearnkit (#81)

* feat: add threshold_summary

* refactor(c2.py): use shorthand type hinting

* feat(sklearnkit): add precision_given_recall_score

* revert: &#34;refactor(c2.py): use shorthand type hinting&#34;

This reverts commit 3e7784ac821371dde6112076e6e25b96998f3d6b.

* test: fix casting ([`a667f0d`](https://github.com/estripling/onekit/commit/a667f0da25c4bf9061496761f44d5830c4654c13))

### Fix

* fix(select_col_types): check for vaild data types (#82)

* fix(select_col_types): check for vaild data types

* fix(select_col_types): type hint ([`36bf1e1`](https://github.com/estripling/onekit/commit/36bf1e10fe81968c44d334b65c0f17a97e90c4b8))

## v1.3.0 (2024-06-03)

### Build

* build: improve setup (#77)

- Replace Makefile with a Python script to run commands
- Enable easy setup on host machine next to docker
- Make sure tests pass one Windows OS

Commits
----------

* build: improve docker setup

* build: refactor clear_cache.py

* build: update poetry.lock

* build: add command.py

* build: update process_argument__create_docs

* docs: update year

* build: add remove_docs

* build(command.py): refactor main

* build(command.py): add poetry commands

* build: update poetry.lock

* revert: &#34;build: update poetry.lock&#34;

This reverts commit 1e1f10cf54790358c5fee4f914a893654fcd277c.

* build: update poetry.lock

* build: delete Makefile

* build: update commands

* build: update commands

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

* build: rename command.py to c2.py ([`7b61657`](https://github.com/estripling/onekit/commit/7b616571789f0c0a962b04d5f8514c58e0377fdc))

* build: update poetry (#74) ([`bae0e62`](https://github.com/estripling/onekit/commit/bae0e6226e08c36d4a245986fe97a40a620c5ba3))

### Feature

* feat: add bool_to_str (#78)

* refactor(bool_to_int): use select_col_types

* feat: add bool_to_str ([`fadd710`](https://github.com/estripling/onekit/commit/fadd710a52ba0f18942796a563017912bef5f595))

* feat: add select_col_types (#76) ([`0720987`](https://github.com/estripling/onekit/commit/0720987349a5c70bc99d3b116796d09905f00024))

* feat: extend digitscale (#75)

* feat(digitscale): add kind

* feat(digitscale): add linear kind

* style: be consistent ([`80771d3`](https://github.com/estripling/onekit/commit/80771d3b865bc1a3d26901726209bd7009b2d25a))

## v1.2.0 (2024-05-10)

### Build

* build: update github workflows (#73)

* build: update github workflows

* build: move time-machine dep to testing group ([`efb2f19`](https://github.com/estripling/onekit/commit/efb2f19a0dd71e724b1526bc3b214729bd373db7))

* build: improve remove-local-branches (#72) ([`0dcd997`](https://github.com/estripling/onekit/commit/0dcd9978cd76e1e9d82efa957fa5830b788c8154))

### Documentation

* docs: update onekit logo (#70) ([`dce595c`](https://github.com/estripling/onekit/commit/dce595c1a148af48d0b500d532367aee20571750))

### Feature

* feat: add digitscale (#69)

* feat(mathkit): add digitscale

* feat(mathkit): add sign

* feat(numpykit): add digitscale

* feat(sparkkit): add with_digitscale

* docs: add mathkit.ipynb ([`4f9c908`](https://github.com/estripling/onekit/commit/4f9c908aa29d20d3b04c18127593b1d86f9a8b18))

### Style

* style: change wording (#71) ([`639d914`](https://github.com/estripling/onekit/commit/639d914225f2a751fbc368acd438a1c6eee0c795))

## v1.1.1 (2024-05-06)

### Fix

* fix: dependencies (#68)

* build: add pytz

* fix: dependencies ([`cb34fc7`](https://github.com/estripling/onekit/commit/cb34fc7cc627e90a5add59404d02e2236f716cec))

### Refactor

* refactor(filter_date): n=inf for no lower bound (#66) ([`46e3606`](https://github.com/estripling/onekit/commit/46e3606739f91d4eee995cc56a57b576a4bf04b8))

### Test

* test: refactor spark session (#67) ([`dc34fe7`](https://github.com/estripling/onekit/commit/dc34fe7d37de8967f034686924472e89bf8321ae))

## v1.1.0 (2024-04-26)

### Feature

* feat(stopwatch): add timezone (#65)

* docs: use consistent style for optional parameters

* refactor(archive_files): add timezone

* feat(stopwatch): add timezone ([`d261904`](https://github.com/estripling/onekit/commit/d2619049e437f6cca9032d82046a6b7247134f90))

* feat: add timestamp (#64)

* feat(pythonkit): add timestamp

* docs(LICENSE): update year

* refactor(timestamp): no positional arguments ([`67e657d`](https://github.com/estripling/onekit/commit/67e657d19ea5f20afaafe276cfd159cbe8f81ee4))

### Refactor

* refactor(filter_date): use with_date_diff_ago (#63) ([`b0bd1ec`](https://github.com/estripling/onekit/commit/b0bd1ecd4b0e6d8c51e12854d6d10601dc1b8882))

## v1.0.0 (2024-04-01)

### Breaking

* refactor: rename sk.daterange -&gt; sk.date_range (#61)

* refactor: rename sk.daterange -&gt; sk.date_range

* refactor(with_weekday): add type hint for determine_weekday

BREAKING CHANGE: rename sk.daterange -&gt; sk.date_range ([`c1b8661`](https://github.com/estripling/onekit/commit/c1b8661025fbf7bab6be4ea682c436dff330ed98))

* refactor: rename daterange -&gt; date_range (#60)

BREAKING CHANGE: rename daterange -&gt; date_range ([`547d5ac`](https://github.com/estripling/onekit/commit/547d5ac9a040843a945e76cb172da503fae7f3c8))

* feat: datecount -&gt; date_count_backward, date_count_forward (#59)

BREAKING CHANGE: split datecount into date_count_backward and date_count_forward ([`fff3c77`](https://github.com/estripling/onekit/commit/fff3c77ef58c1ba7e96f430e6b78d48d15aebf20))

* feat: improve date functionality (#54)

* feat: add filter_date

* refactor: rename day -&gt; d

* refactor: rename daycount -&gt; datecount

* refactor: rename n_days -&gt; number_of_days

* test(relative_date): rename days -&gt; dates

* feat: add date_ahead

* feat: add date_ago

* fix: remove relative_date

* refactor: change order

BREAKING CHANGE: renamed date functions ([`72fba47`](https://github.com/estripling/onekit/commit/72fba4706626ff8b18e208e86271e9595ebe1962))

### Feature

* feat: add with_increasing_id (#62)

* feat: add with_increasing_id

* test: fix nondeterministic test ([`1dd4c19`](https://github.com/estripling/onekit/commit/1dd4c19512d8135f67bc4b18d00d3911ef9767ac))

* feat: add bool_to_int (#58)

* feat: add bool_to_int

* fix(peek): use bool_to_int before converting to pandas ([`bbde5aa`](https://github.com/estripling/onekit/commit/bbde5aa04fadb42de40e2d46d85e0300180d70a0))

* feat: split with_date_diff into with_date_diff_ago and with_date_diff_ahead (#57) ([`1175784`](https://github.com/estripling/onekit/commit/11757848773a9378e89bcc9bb5d3c5b535105a33))

* feat: add date diff (#55)

* docs(str_to_col): update docstring

* feat: add with_date_diff ([`2504dcd`](https://github.com/estripling/onekit/commit/2504dcdf86a77454208c1bff4647ed7c1bc20b9c))

* feat: add all_col and any_col (#52)

* feat: add all_col

* feat: add any_col ([`fb0c07a`](https://github.com/estripling/onekit/commit/fb0c07a55e30e1db6d598b7a47cb0997cfa4a47b))

### Style

* style: change type hint for star parameters (#53) ([`840bc66`](https://github.com/estripling/onekit/commit/840bc66e371be24b254fcc87b1836e0c57b968aa))

### Test

* test: refactor to have shorter tests (#56)

* test: refactor test_with_endofweek_date

* test: refactor test_with_index

* test: refactor test_with_startofweek_date

* test: refactor test_with_weekday

* test: refactor test_filter_date ([`7aa966d`](https://github.com/estripling/onekit/commit/7aa966d409498c03b2b85b2552d8b6952105fd1b))

## v0.16.0 (2024-03-03)

### Feature

* feat: add check column functionality (#51)

* feat: add check_column_present

* feat: add has_column ([`639f379`](https://github.com/estripling/onekit/commit/639f379429ac0cfaeaea821190649ac7782aa8ce))

### Refactor

* refactor: rename check functions (#50)

* refactor: rename check_dataframe_equal -&gt; assert_dataframe_equal

* refactor: rename check_row_count_equal -&gt; assert_row_count_equal

* refactor: rename check_row_equal -&gt; assert_row_equal

* refactor: rename check_schema_equal -&gt; assert_schema_equal

* refactor(example.ipynb): update notebook

* build: add pre-commit hook ([`0bd7103`](https://github.com/estripling/onekit/commit/0bd7103ff176e5c3645a8f3596ffebaa779d1831))

## v0.15.0 (2024-01-29)

### Feature

* feat(pandaskit): add cvf (#49) ([`a8c9cb6`](https://github.com/estripling/onekit/commit/a8c9cb66e87f1ff3ef968a6604be96c110f23151))

## v0.14.0 (2024-01-24)

### Feature

* feat: improve pandaskit.profile (#48)

* refactor(pandaskit.profile): no special characters

* test(pandaskit.profile): add isnull and notull query

* feat(numpykit): add quantile argument

* test: rename function

* refactor(pandaskit.profile): use basic_info_df variable

* feat(pandaskit.profile): add sum ([`f10bf9a`](https://github.com/estripling/onekit/commit/f10bf9a0acf4c981dcd1e5263a2a2afbee28de81))

## v0.13.0 (2023-12-22)

### Feature

* feat: add numpykit (#47)

* refactor(check_vector): move optfunckit -&gt; numpykit

* feat(numpykit): add stderr ([`d580b8f`](https://github.com/estripling/onekit/commit/d580b8f3167731c028dc65c62b7b76f46e7d259e))

### Refactor

* refactor: move math functions to mathkit (#46)

* refactor(collatz): move pythonkit -&gt; mathkit

* refactor(fibonacci): move pythonkit -&gt; mathkit

* refactor: move isdivisible iseven isodd to mathkit ([`c4a5bd8`](https://github.com/estripling/onekit/commit/c4a5bd8c83986b626d6c7d2b01d6ee920b746263))

* refactor: small updates (#45)

* refactor(XyzPoints): adjust type hints

* docs(peaks): add reference

* docs(optfunckit): add ai-roomi reference

* docs(concat_strings): add docstring example ([`c6f02d7`](https://github.com/estripling/onekit/commit/c6f02d701965b6c34ae50392f5b2fdd5a573d654))

## v0.12.0 (2023-12-21)

### Feature

* feat: add vizkit (#44)

* build: add matplotlib dependency

* feat(pythonkit): add op

* feat: add vizkit ([`5f5e824`](https://github.com/estripling/onekit/commit/5f5e82466d69673c21c3e436787f0005de21bc3b))

* feat: add fbench functions (#43)

* feat: add test functions for minimization

* feat(optfunckit): add negate

* feat(optfunckit): add bump

* refactor(sinc): negate

* style(optfunckit): correct indentation

* docs(peaks): add reference in docstring

* docs(negate): add maximization example ([`173052f`](https://github.com/estripling/onekit/commit/173052f102696f8cb2f0652c7e8e19875f43556e))

### Refactor

* refactor: error messages (#42) ([`e7e353e`](https://github.com/estripling/onekit/commit/e7e353e234b8ebd32ce2b036e51122033e5c8837))

## v0.11.0 (2023-12-11)

### Feature

* feat: add spark functions (#41)

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

* feat(sparkkit): add evaluation functions ([`d421ac4`](https://github.com/estripling/onekit/commit/d421ac40170bf0c2d9222d1cc7cab618a73d7678))

* feat: add pandaskit (#39)

* feat(pandaskit): add union

* feat(pandaskit): add join

* test: refactor pandas test_union

* feat(pandaskit): add profile ([`60e9a09`](https://github.com/estripling/onekit/commit/60e9a094cc38e05566609ed31d6d87934bf27663))

### Refactor

* refactor: rename num variables + type hints (#40)

* refactor: int values as n

* test: add type hints ([`6269e3a`](https://github.com/estripling/onekit/commit/6269e3a38ab07b9cb685cec769b65e3bdb8036fa))

## v0.10.1 (2023-11-22)

### Documentation

* docs: update onekit author and description (#38)

* docs: update license

* build: update pyproject.toml

* docs: update README.md

* docs: update author ([`62947fb`](https://github.com/estripling/onekit/commit/62947fb6a8be9711fc55e31cac676697e4326022))

### Fix

* fix(README.md): disclaimer section ([`9428fee`](https://github.com/estripling/onekit/commit/9428feef002ba4648f1c3f63f2de6f5ad7b1fd1d))

## v0.10.0 (2023-11-22)

### Documentation

* docs(sparkkit.peek): add missing shape argument (#36)

* docs(sparkkit.peek): add missing shape argument

* refactor: DfIdentityFunction -&gt; DFIdentityFunc ([`7810dc3`](https://github.com/estripling/onekit/commit/7810dc3e954483ae482c8cda59289513fad6a88b))

### Feature

* feat(sparkkit): add cvf (#37)

* feat(sparkkit): add str_to_col

* refactor: DFIdentityFunc -&gt; SparkDFIdentityFunc

* style(peek): remove docstring of inner function

* test(str_to_col): no parametrize

* feat(sparkkit): add cvf ([`b989f31`](https://github.com/estripling/onekit/commit/b989f31bb4a9fa9a0e40eed24852cf8b7a6b1335))

## v0.9.0 (2023-11-20)

### Breaking

* refactor: rename modules (#33)

* refactor: pytlz -&gt; pythonkit

* refactor: sparktlz -&gt; sparkkit

* build(pyproject.toml): rename sparktlz -&gt; sparkkit

* refactor: pdtlz -&gt; pandaskit

BREAKING CHANGE: rename modules to have kit suffix ([`5dfd157`](https://github.com/estripling/onekit/commit/5dfd157ae6c4ae7e89afe58e121f7dde4d2f719e))

### Feature

* feat(sparkkit): add peek (#35) ([`69b08e9`](https://github.com/estripling/onekit/commit/69b08e90e6399639bf013414a0b6d45d1c008393))

### Refactor

* refactor(signif): curry function (#34) ([`6880f9c`](https://github.com/estripling/onekit/commit/6880f9c297f1e178612b72187528a289ffafc25a))

## v0.8.0 (2023-11-17)

### Documentation

* docs: add notebook examples (#32)

* docs: add examples

* docs(example.ipynb): add highlight_string_differences

* docs(example.ipynb): add stopwatch

* refactor: example.ipynb -&gt; examples.ipynb ([`59c972b`](https://github.com/estripling/onekit/commit/59c972b9507c87bef0fe37012065b69b5632c810))

### Feature

* feat: migrate bumbag time functions (#29)

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

* feat(pytlz): add relative_date ([`692542b`](https://github.com/estripling/onekit/commit/692542beec78adc4ff272472b305451408a6e94e))

* feat: migrate bumbag io functions (#27)

* feat(pytlz): add lazy_read_lines

* feat(pytlz): add prompt_yes_no

* feat(pytlz): add archive_files ([`4fd81a3`](https://github.com/estripling/onekit/commit/4fd81a37296df9e7c71584ad2155e45a6b9bb6b9))

### Refactor

* refactor: isdivisibleby -&gt; isdivisible (#31) ([`bb9af5a`](https://github.com/estripling/onekit/commit/bb9af5a4893c367f36acd85b34c92751242bc8b6))

* refactor(relative_date): change signature (#30) ([`049ab7d`](https://github.com/estripling/onekit/commit/049ab7d9745f2688e383dc2607202f11908aca42))

### Test

* test: add itertools (#28) ([`b83469d`](https://github.com/estripling/onekit/commit/b83469d1a05c139e8ff6f941119e805ba4dca4a1))

## v0.7.0 (2023-11-17)

### Documentation

* docs: update docstrings (#24)

* docs(reduce_sets): update docstring example

* docs(source_code): update docstring ([`1585554`](https://github.com/estripling/onekit/commit/15855544eb2b85abe55ff34908028e9c10d45a54))

### Feature

* feat: migrate bumbag string functions (#26)

* feat(pytlz): add concat_strings

* feat(pytlz): add create_path

* feat(pytlz): add filter_regex

* refactor: use iterable instead of sequence

* feat(pytlz): add map_regex

* feat(pytlz): add headline

* feat(pytlz): add remove_punctuation

* feat(pytlz): add highlight_string_differences ([`95b8e7d`](https://github.com/estripling/onekit/commit/95b8e7dc4b405c58b49a021fea26f2ac20a87f07))

* feat(pytlz): add are_predicates_true (#25)

- Apply DRY principle: replace all_predicates_true and any_predicates_true ([`de96017`](https://github.com/estripling/onekit/commit/de960173809febc8947b80e2a7735ae140574fd7))

## v0.6.0 (2023-11-15)

### Documentation

* docs(pytlz): rephrase docstring of bool functions (#20) ([`fa8a1df`](https://github.com/estripling/onekit/commit/fa8a1df96aaf72e77f9462c0ac4bd161784d4ac0))

### Feature

* feat: migrate bumbag random functions (#22)

* feat(pytlz): add check_random_state

* feat(pytlz): add coinflip

* test: add raises checks in else clause

* docs(coinflip): add docstring example

* docs(coinflip): add docstring example with biased coin ([`8906f15`](https://github.com/estripling/onekit/commit/8906f15047a57970b88eb61aef52e20a6210a60b))

* feat(pytlz): add collatz and fibonacci (#21)

* feat(pytlz): add collatz

* feat(pytlz): add fibonacci

* style(collatz): update references in docstring ([`0881625`](https://github.com/estripling/onekit/commit/088162506219ef5a95acb3dbde81dd108d686071))

### Refactor

* refactor: curry functions only where necessary (#23)

* refactor(extend_range): replace curry with partial

* docs(isdivisibleby): indicate function is curried

* docs(reduce_sets): indicate function is curried

* refactor(signif): replace curry with partial ([`acbbcea`](https://github.com/estripling/onekit/commit/acbbcea2f31f063469cb140ea70f18d63cbea099))

## v0.5.0 (2023-11-14)

### Build

* build(pyproject.toml): update classifiers (#14) ([`6bdc390`](https://github.com/estripling/onekit/commit/6bdc3906c552caa12336aafa7f8a4f4020022e40))

### Ci

* ci(release.yml): use release token (#12) ([`f2ca10a`](https://github.com/estripling/onekit/commit/f2ca10a65ca2e3828685148a2a96f3934afe51cd))

### Documentation

* docs: update developer guide (#17) ([`3d141b5`](https://github.com/estripling/onekit/commit/3d141b528a2823c555e6b8c364707f3291e5856f))

* docs(README.md): remove example (#13) ([`9f510ac`](https://github.com/estripling/onekit/commit/9f510ace488f6a77b1b482432afc09204e82e2d2))

### Feature

* feat: add bumbag core functions (#18)

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

* docs(signif): add type call to show it is curried ([`50badd8`](https://github.com/estripling/onekit/commit/50badd8d105adc8c9d7acc94f70ef0fcffaa8af1))

### Refactor

* refactor: predicate functions (#19)

* refactor(all_predicate_true): use inner function

* refactor(any_predicate_true): use inner function

* docs(isdivisibleby): update docstring for consistency ([`2a9f699`](https://github.com/estripling/onekit/commit/2a9f699d6c78b8f91a2a762f74ae7935a75c8ce2))

### Style

* style: update docs (#11)

* docs: update module docstring

* refactor(Makefile): add missing phony

* docs(pytlz.flatten): update type hinting

* docs(sparktlz.union): update type hinting

* refactor: rename SparkDataFrame -&gt; SparkDF ([`2527c1c`](https://github.com/estripling/onekit/commit/2527c1c6bc5bdfdc87b2bbc913bf2585e782daaf))

### Test

* test: ignore Spark doctest (#16) ([`e682a09`](https://github.com/estripling/onekit/commit/e682a09f86a595b1549356322dc725e99092ff84))

### Unknown

* revert: &#34;ci(release.yml): use release token (#12)&#34; (#15)

This reverts commit f2ca10a65ca2e3828685148a2a96f3934afe51cd. ([`0054460`](https://github.com/estripling/onekit/commit/0054460595c40699e8bda73576c04ac8281dca3d))

## v0.4.0 (2023-11-13)

### Build

* build: add pandas (#9) ([`a5185ba`](https://github.com/estripling/onekit/commit/a5185baf15be09815fed92c34a70191ebdc86a91))

### Documentation

* docs: add example (#10)

* docs(example.ipynb): add pytlz.flatten and sparktlz.union

* style(sparktlz.union): rearrange docsting imports

* docs(README.md): add example ([`9af41f3`](https://github.com/estripling/onekit/commit/9af41f347198d43d42def455cf6e2076992fe603))

* docs: import pytlz directly (#6)

* refactor: import pytlz directly

* docs: no import onekit as ok

* docs(README.md): rename example usage to examples

* docs: add module description ([`e777ec4`](https://github.com/estripling/onekit/commit/e777ec440be757da0cd7a896563580f8d343667a))

### Feature

* feat(sparktlz): add union (#8) ([`7680762`](https://github.com/estripling/onekit/commit/76807626f3c64755171654d43749eb9c875c9d8c))

* feat(pytlz): add flatten (#7)

* tests(date_to_str): correct test name

* feat(pytlz): add flatten ([`7629260`](https://github.com/estripling/onekit/commit/7629260f14a0ce7af3afa78e624753f03efd55e1))

* feat: add sparktlz (#5)

* build: add pyspark

* tests: set up Spark session ([`443dc33`](https://github.com/estripling/onekit/commit/443dc338543f301da19d4a656e65bb7632949363))

## v0.3.0 (2023-11-09)

### Feature

* feat: add date_to_str (#3)

* num_to_str: rephrase docstring

* pytlz: add date_to_str ([`1068a65`](https://github.com/estripling/onekit/commit/1068a65df00a9c89997f3fe8c216e4e03f818e59))

### Refactor

* refactor: import onekit as ok (#4) ([`aa077f8`](https://github.com/estripling/onekit/commit/aa077f818ed942d6f7742ee79d08e435af893a19))

* refactor: num_to_str (#2)

* num_to_str: improve type hinting

* changelog.md: remove bullet points ([`2447205`](https://github.com/estripling/onekit/commit/2447205871da761d1ac321475fac4bea1ee6107e))

## v0.2.0 (2023-11-09)

### Ci

* ci: GitHub release before PyPI ([`1199208`](https://github.com/estripling/onekit/commit/11992089d2396ef24f81bfa2c1e5624750c8b3b5))

### Feature

* feat(pytlz): add num_to_str (#1)

* feat(pytlz): add num_to_str

* docs: show pytlz module ([`cc130e6`](https://github.com/estripling/onekit/commit/cc130e6850d9a8a325067a0935ccceada27b0e40))

## v0.1.0 (2023-11-09)

### Feature

* feat: add repository setup ([`2603f02`](https://github.com/estripling/onekit/commit/2603f02007d10a8b932a0510495df89a9c64635b))
