
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)



def close(session_attributes, fulfillment_state, message):
    response = {
        'sessionAttributes': session_attributes,
        'dialogAction': {
            'type': 'Close',
            'fulfillmentState': fulfillment_state,
            'message': message
        }
    }

    return response

""" --- Functions that control the bot's behavior --- """


def get_balance(intent_request):
	logger.debug('in get_balance')
	session_attributes = intent_request['sessionAttributes'] if intent_request['sessionAttributes'] is not None else {}
	slots = intent_request['currentIntent']['slots']
	account_type = slots['AccountType']
	balance = 0
	if account_type.lower() == 'checking':
		balance = 5000
	if account_type.lower() == 'saving':
		balance = 10000
	
	session_attributes['currentIntent'] = intent_request['currentIntent']['name']

	return close(
		session_attributes,
        'Fulfilled',
        {
            'contentType': 'PlainText',
            'content': 'You account balance is ${}.'.format(balance)
        }
	)

def get_loan_balance(intent_request):
	logger.debug('in get_loan_balance')
	
	session_attributes = intent_request['sessionAttributes'] if intent_request['sessionAttributes'] is not None else {}
	slots = intent_request['currentIntent']['slots']
	loan_type = slots['LoanType']
	balance = 0
	rate = 0
	
	if loan_type.lower() == 'car':
		balance = 35000
		rate = 0.05
	if loan_type.lower() == 'home':
		balance = 1200000
		rate = 0.035
		
	session_attributes['currentIntent'] = intent_request['currentIntent']['name']
	session_attributes['loan_type'] = loan_type
	return close(
		session_attributes,
        'Fulfilled',
        {
            'contentType': 'PlainText',
            'content': "You {} balance is ${}, and your current rate is {:.2%}.  You might qualify for lower rates, please type 'more loan info' to hear more?".format(loan_type, balance, rate)
        }
	)
	
def get_loan_offer(intent_request):
	logger.debug('in get_loan_balance')
	
	session_attributes = intent_request['sessionAttributes'] if intent_request['sessionAttributes'] is not None else {}
	slots = intent_request['currentIntent']['slots']
	
	loan_type = slots['LoanType']
	

	if loan_type is None and session_attributes['currentIntent'] == 'GetLoanDetail':
		loan_type = session_attributes['loan_type']
	
	
	rate = 0
	
	if loan_type.lower() == 'car':
		rate = 0.04
	if loan_type.lower() == 'home':
		rate = 0.03
	
	session_attributes['currentIntent'] = intent_request['currentIntent']['name']
	
	return close(
		session_attributes,
        'Fulfilled',
        {
            'contentType': 'PlainText',
            'content': 'The current rate for {} loan is {:.2%}'.format(loan_type.lower(), rate)
        }
	)


# --- Intents ---


def dispatch(intent_request):
    """
    Called when the user specifies an intent for this bot.
    """

    logger.debug('dispatch userId={}, intentName={}'.format(intent_request['userId'], intent_request['currentIntent']['name']))

    intent_name = intent_request['currentIntent']['name']

    
    # Dispatch to your bot's intent handlers

    if intent_name == 'GetAccountDetail':
    	return get_balance(intent_request)
    elif intent_name == 'GetLoanDetail':
        return get_loan_balance(intent_request)
    elif intent_name == 'GetLoanProducts':
    	return get_loan_offer(intent_request)
    

    raise Exception('Intent with name ' + intent_name + ' not supported')


# --- Main handler ---


def lambda_handler(event, context):
    """
    Route the incoming request based on intent.
    The JSON body of the request is provided in the event slot.
    """

    logger.debug('event.bot.name={}'.format(event['bot']['name']))

    return dispatch(event)

