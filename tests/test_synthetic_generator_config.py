from src.data.synthetic_generator import GeneratorConfig


def test_generator_config_accepts_constructor_arguments():
    config = GeneratorConfig(num_records=1000, num_entities=200, seed=7)

    assert config.num_records == 1000
    assert config.num_entities == 200
    assert config.seed == 7
