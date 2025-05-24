#!/usr/bin/env python3
"""
Main script to demonstrate the configuration system usage.
"""

import sys
import os
from config import ConfigReader, run_fact_check_experiment

def main():
    """
    Main function
    """
    # You can specify a different config file path if needed
    config_file = "config.yml"

    # Check if config file exists
    if not os.path.exists(config_file):
        print(f"❌ Configuration file '{config_file}' not found!")
        print("Please create a config.yml file in the current directory.")
        sys.exit(1)

    try:
        # Initialize the configuration reader
        config_reader = ConfigReader(config_file)

        # Load the configuration
        config = config_reader.load_config()

        if not config:
            print("❌ Failed to load configuration!")
            sys.exit(1)

        # Validate the configuration
        validation_result = config_reader.validate_config()

        # Display validation results
        config_reader.print_validation_results()

        # Display the configuration beautifully
        config_reader.print_configuration()

        # Only proceed if configuration is valid
        if validation_result.is_valid:
            print("\n✅ Configuration is valid! Proceeding with experiment...")

            # Run the fact-check experiment with the validated configuration
            run_fact_check_experiment(config)

        else:
            print("\n❌ Configuration validation failed!")
            print("Please fix the errors and warnings before running the experiment.")

            # Print specific guidance
            if validation_result.errors:
                print("\n🔧 To fix errors:")
                for error in validation_result.errors:
                    print(f"   • {error}")

            if validation_result.warnings:
                print("\n⚠️  Warnings to consider:")
                for warning in validation_result.warnings:
                    print(f"   • {warning}")

            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n🛑 Process interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()