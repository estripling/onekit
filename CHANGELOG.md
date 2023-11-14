# CHANGELOG



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
