def set_template(args):
    # Set the templates here
    
    if args.template.find('AIN') >=0:
        args.model = 'AIN'

    if args.template.find('AIN2') >= 0:
        args.model = 'AIN2'
