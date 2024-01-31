"""Test build rules for `Dinosaur`."""

load("//devtools/build_cleaner/skylark:build_defs.bzl", "register_extension_info")
load("//devtools/python/blaze:pytype.bzl", "pytype_strict_binary")
load("//devtools/python/blaze:strict.bzl", "py_strict_binary", "py_strict_test")

CONFIGURATIONS = {
    "cpu": struct(
        backend = "cpu",
        tags = [],
        deps = [],
    ),
    "gpu": struct(
        backend = "gpu",
        tags = ["requires-gpu-nvidia"],
        deps = ["//learning/brain/research/jax:gpu_support"],
    ),
    "tpu": struct(
        backend = "tpu",
        tags = ["requires-dragonfish"],
        deps = ["//learning/brain/research/jax:tpu_support"],
    ),
}

def dinosaur_test(
        name,
        srcs,
        deps = [],
        args = [],
        configs = ["cpu"],
        data = [],
        tags = [],
        main = None,
        shard_count = None,
        size = None):
    """Defines a set of tests that run on CPU and TPU.

    Generates test targets named:
    :name  # test suite that tests all backends.
    :name_cpu
    :name_tpu

    By default, :name_cpu (if "cpu" is not in `configs`) and :name_tpu  (if "tpu" is not
    in `configs`) are manual tests that don't run on TAP. These targets are useful for local
    debugging with `--pdb`.

    Args:
      name: test name
      srcs: source files
      deps: dependencies
      args: arguments to pass to the test
      configs: which configurations to run tests with. Must be a nonempty subset of ["cpu", "tpu"].
          Defaults to "cpu".
      data: data dependencies
      tags: test tags
      main: the Python main file.
      shard_count: a dictionary keyed by backend name of per-backend shard counts.
      size: the test size.
    """
    if main == None:
        if len(srcs) == 1:
            main = srcs[0]
        else:
            fail("Only one test source file is currently supported.")

    # Deps and tags that apply to all targets.
    all_configs = [
        "cpu",
        "tpu",
    ]
    all_test_deps = [
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/jax",
    ]
    all_test_tags = [
        "optonly",
    ]
    unlisted_config_tags = [
        "manual",
        "notap",
    ]

    enabled_tests = []
    for config_name in all_configs:
        config = CONFIGURATIONS.get(config_name)
        if not config:
            fail("Unknown config: {}".format(config_name))
        config_args = [
            "--jax_platform_name=" + config.backend,
        ]
        config_tags = config.tags + ["backend_{}".format(config.backend)]
        test_name = "{}_{}".format(name, config_name)
        enabled_tests.append(test_name)

        current_tags = tags + config_tags + all_test_tags
        if config_name not in configs:
            current_tags += unlisted_config_tags

        py_strict_test(
            name = test_name,
            main = main,
            srcs = srcs,
            data = data,
            args = args + config_args,
            deps = deps + config.deps + all_test_deps,
            tags = current_tags,
            python_version = "PY3",
            srcs_version = "PY3",
            shard_count = shard_count.get(config_name, None) if shard_count else None,
            size = size,
        )

    native.test_suite(name = name, tests = enabled_tests, tags = ["-manual"])

def dinosaur_binary(
        name,
        srcs,
        deps = [],
        args = [],
        configs = ["cpu"],
        data = [],
        main = None,
        enforce_typing = False):
    """Defines a binary for CPU or TPU

    Generates binary targets named:
    :name_cpu  # if "cpu" is in `configs`.
    :name_gpu  # if "gpu" is in `configs`.
    :name_tpu  # if "tpu" is in `configs`.

    Args:
      name: binary name
      srcs: source files
      deps: dependencies
      args: arguments to pass to the binary
      configs: which configurations to run binaries with. Must be a nonempty subset of ["cpu", "gpu", "tpu"]. Defaults to "cpu".
      data: data dependencies
      main: the Python main file.
      enforce_typing: Whether to check typing.
    """
    if main == None:
        if len(srcs) == 1:
            main = srcs[0]
        else:
            fail("Only one source file is currently supported.")
    all_binary_deps = [
        "//third_party/py/jax",
    ]
    mk_binary = pytype_strict_binary if enforce_typing else py_strict_binary
    for config_name in configs:
        config = CONFIGURATIONS.get(config_name)
        if not config:
            fail("Unknown config: {}".format(config_name))
        config_name = "{}_{}".format(name, config_name)
        mk_binary(
            name = config_name,
            main = main,
            srcs = srcs,
            data = data,
            args = args,
            deps = deps + config.deps + all_binary_deps,
            python_version = "PY3",
            srcs_version = "PY3",
            test_lib = True,
        )

# go/build-cleaner-build-extensions
register_extension_info(
    extension = dinosaur_test,
    label_regex_for_dep = "{extension_name}_[ct]pu",
)
register_extension_info(
    extension = dinosaur_binary,
    label_regex_for_dep = "{extension_name}_[ct]pu",
)
