from pathlib import Path


def iterate_variants(parent_path: Path, up_to: Path) -> list[list[Path]]:
    def return_variant_and_default(path: Path):
        default_variant = path / "default.yaml"
        variants = list(path.glob("*.yaml"))
        if default_variant.exists() and len(variants) > 1:
            for variant in variants:
                if variant != default_variant:
                    yield [default_variant, variant]
            yield [default_variant]
        else:
            yield list(path.glob("*.yaml"))

    if parent_path == up_to:
        return list(return_variant_and_default(parent_path))
    result = []
    for variant in iterate_variants(parent_path.parent, up_to):
        current_variations = return_variant_and_default(parent_path)
        for path in current_variations:
            result.append(variant + path)
    return result


launch_config = """
{
    // Verwendet IntelliSense zum Ermitteln möglicher Attribute.
    // Zeigen Sie auf vorhandene Attribute, um die zugehörigen Beschreibungen anzuzeigen.
    // Weitere Informationen finden Sie unter https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Refresh launch.json",
            "type": "python",
            "request": "launch",
            "program": "scripts/create_vs_launch_config.py",
            "console": "integratedTerminal",
        },
"""
config_path = Path("configs")
variants_path = config_path / "variants"
paths = list(variants_path.glob("**/*.yaml"))

for path in paths:
    variants_configs = [x / "default.yaml" for x in path.parents]
    variants_configs = list(filter(lambda x: x.exists(), variants_configs))
    if variants_configs[0] != path:
        variants_configs = [path] + variants_configs
    variants_configs.reverse()

    for env in config_path.glob("*"):
        if env == variants_path:
            continue

        env_path = env / path.relative_to(variants_path)
        env_path = [parent for parent in env_path.parents if parent.exists()][0]
        env_variants = iterate_variants(env_path, env)

        for env_variants_config in env_variants:
            experiment_name = "Train "
            variant_name = " - ".join(
                [
                    x.stem if x.stem != "default" else x.parent.stem
                    for x in list(reversed(variants_configs))
                    if x.parent.stem != "variants" or x.stem != "default"
                ]
             )
            experiment_name += variant_name
            experiment_name += " on "
            environment_name = " - ".join(
                [
                    x.stem if x.stem != "default" else x.parent.stem
                    for x in env_variants_config
                    if x.parent.stem != "variants" or x.stem != "default"
                ]
            )
            experiment_name += environment_name
            launch_config += f"""
                {{
                    "name": "{experiment_name}",
                    "type": "python",
                    "request": "launch",
                    "program": "cli.py",
                    "console": "integratedTerminal",
                    "justMyCode": false,
                    "args": [
                        "--config", {', "--config", '.join([f'"{str(x)}"' for x in variants_configs + env_variants_config])}, "--main.algorithm.init_args.tensorboard_log", "logs/{environment_name.replace(" - ", "/")}", "--learn.tb_log_name", "{variant_name}", "train"
                    ]
                }},
            """

launch_config += """
    ]
}
"""

with open(".vscode/launch.json", "w") as f:
    f.write(launch_config)
